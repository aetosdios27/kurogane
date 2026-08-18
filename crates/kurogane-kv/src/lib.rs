//! Replicated in-memory key/value state machine on top of `kurogane-raft`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use kurogane_raft::{Effect, Event, Node};

/// A client operation against the replicated key/value state machine.
/// `Get` is a command, not a side-channel read: routing it through the log
/// is what makes it linearizable without a heartbeat-lease mechanism.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    Set { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Get { key: Vec<u8> },
}

/// Invalid or truncated command bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    Truncated,
    UnknownTag(u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => {
                formatter.write_str("command bytes ended before a complete value was read")
            }
            Self::UnknownTag(tag) => write!(formatter, "unknown command tag {tag}"),
        }
    }
}

impl Error for DecodeError {}

const SET_TAG: u8 = 0;
const DELETE_TAG: u8 = 1;
const GET_TAG: u8 = 2;

impl Command {
    /// Encodes this command as a self-describing byte string: a one-byte
    /// tag followed by `u32`-length-prefixed byte strings. Hand-rolled
    /// rather than pulling in `serde`, matching the project's existing
    /// zero-dependency style; a real wire format arrives with milestone six.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            Self::Set { key, value } => {
                bytes.push(SET_TAG);
                encode_bytes(&mut bytes, key);
                encode_bytes(&mut bytes, value);
            }
            Self::Delete { key } => {
                bytes.push(DELETE_TAG);
                encode_bytes(&mut bytes, key);
            }
            Self::Get { key } => {
                bytes.push(GET_TAG);
                encode_bytes(&mut bytes, key);
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut reader = ByteReader::new(bytes);
        match reader.read_u8()? {
            SET_TAG => {
                let key = reader.read_bytes()?;
                let value = reader.read_bytes()?;
                Ok(Self::Set { key, value })
            }
            DELETE_TAG => Ok(Self::Delete {
                key: reader.read_bytes()?,
            }),
            GET_TAG => Ok(Self::Get {
                key: reader.read_bytes()?,
            }),
            other => Err(DecodeError::UnknownTag(other)),
        }
    }
}

fn encode_bytes(buf: &mut Vec<u8>, value: &[u8]) {
    buf.extend_from_slice(&(value.len() as u32).to_be_bytes());
    buf.extend_from_slice(value);
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or(DecodeError::Truncated)?;
        self.position += 1;
        Ok(byte)
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, DecodeError> {
        let length_bytes = self
            .bytes
            .get(self.position..self.position + 4)
            .ok_or(DecodeError::Truncated)?;
        let length = u32::from_be_bytes(length_bytes.try_into().expect("checked length")) as usize;
        self.position += 4;

        let value = self
            .bytes
            .get(self.position..self.position + length)
            .ok_or(DecodeError::Truncated)?;
        self.position += length;
        Ok(value.to_vec())
    }
}

/// The outcome of applying one command, kept around for later retrieval —
/// see `Replica::applied_result`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ApplyResult {
    Set { previous: Option<Vec<u8>> },
    Delete { previous: Option<Vec<u8>> },
    Get { value: Option<Vec<u8>> },
}

/// The replicated key/value map. Owns only in-memory state; persistence is
/// milestone five's job.
#[derive(Debug, Default)]
pub struct StateMachine {
    entries: BTreeMap<Vec<u8>, Vec<u8>>,
    last_applied: u64,
}

impl StateMachine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn last_applied(&self) -> u64 {
        self.last_applied
    }

    pub fn get(&self, key: &[u8]) -> Option<&[u8]> {
        self.entries.get(key).map(Vec::as_slice)
    }

    /// Applies `command` as the next entry in order, advancing
    /// `last_applied` by exactly one. The caller owns index bookkeeping
    /// (see `Replica`) — this trusts it, rather than re-validating an
    /// invariant only one caller can ever violate.
    pub fn apply(&mut self, command: &Command) -> ApplyResult {
        self.last_applied += 1;
        match command {
            Command::Set { key, value } => {
                let previous = self.entries.insert(key.clone(), value.clone());
                ApplyResult::Set { previous }
            }
            Command::Delete { key } => {
                let previous = self.entries.remove(key);
                ApplyResult::Delete { previous }
            }
            Command::Get { key } => ApplyResult::Get {
                value: self.entries.get(key).cloned(),
            },
        }
    }
}

/// Wraps a `Node`, automatically draining newly committed log entries into
/// a `StateMachine` as they land — proposal acceptance (`propose`'s return
/// value) and application/result delivery (`applied_result`) stay distinct.
#[derive(Debug)]
pub struct Replica {
    node: Node,
    state_machine: StateMachine,
    results: BTreeMap<u64, ApplyResult>,
}

impl Replica {
    pub fn new(node: Node) -> Self {
        Self {
            node,
            state_machine: StateMachine::new(),
            results: BTreeMap::new(),
        }
    }

    pub fn node(&self) -> &Node {
        &self.node
    }

    pub fn state_machine(&self) -> &StateMachine {
        &self.state_machine
    }

    /// The outcome of the command applied at `index`, if that index has
    /// been applied yet. `None` both before commit and in the gap between
    /// commit and application.
    pub fn applied_result(&self, index: u64) -> Option<&ApplyResult> {
        self.results.get(&index)
    }

    /// Proposes `command` if this replica's node is the leader, returning
    /// its log index. Never returns the eventual result -- the entry may
    /// not even commit -- that arrives later through `applied_result`.
    pub fn propose(&mut self, command: Command) -> Option<u64> {
        let index = self.node.propose(command.encode())?;
        self.drain_committed();
        Some(index)
    }

    pub fn step(&mut self, event: Event) -> Vec<Effect> {
        let effects = self.node.step(event);
        self.drain_committed();
        effects
    }

    fn drain_committed(&mut self) {
        while self.state_machine.last_applied() < self.node.commit_index() {
            let index = self.state_machine.last_applied() + 1;
            let entry = &self.node.log()[(index - 1) as usize];
            let command = Command::decode(&entry.command)
                .expect("this replica only ever proposes commands it encoded itself");
            let result = self.state_machine.apply(&command);
            self.results.insert(index, result);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use kurogane_raft::{Message, Node, NodeId, Role};

    use super::{ApplyResult, Command, DecodeError, Effect, Event, Replica, StateMachine};

    #[test]
    fn round_trips_set() {
        let command = Command::Set {
            key: vec![1, 2, 3],
            value: vec![9, 9],
        };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_delete() {
        let command = Command::Delete { key: vec![7] };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_get() {
        let command = Command::Get { key: vec![4, 5] };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_empty_keys_and_values() {
        let command = Command::Set {
            key: Vec::new(),
            value: Vec::new(),
        };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn rejects_an_unknown_tag() {
        assert_eq!(Command::decode(&[99]), Err(DecodeError::UnknownTag(99)));
    }

    #[test]
    fn rejects_empty_bytes() {
        assert_eq!(Command::decode(&[]), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_a_truncated_length_prefix() {
        assert_eq!(Command::decode(&[0, 0, 0]), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_a_truncated_value() {
        // Tag 0 (Set), key length 5, but only 2 key bytes follow.
        assert_eq!(
            Command::decode(&[0, 0, 0, 0, 5, 1, 2]),
            Err(DecodeError::Truncated)
        );
    }

    #[test]
    fn set_returns_previous_value_and_stores_the_new_one() {
        let mut state_machine = StateMachine::new();

        let first = state_machine.apply(&Command::Set {
            key: vec![1],
            value: vec![10],
        });
        assert_eq!(first, ApplyResult::Set { previous: None });
        assert_eq!(state_machine.get(&[1]), Some(&[10][..]));

        let second = state_machine.apply(&Command::Set {
            key: vec![1],
            value: vec![20],
        });
        assert_eq!(
            second,
            ApplyResult::Set {
                previous: Some(vec![10])
            }
        );
        assert_eq!(state_machine.get(&[1]), Some(&[20][..]));
    }

    #[test]
    fn delete_returns_and_removes_the_previous_value() {
        let mut state_machine = StateMachine::new();
        state_machine.apply(&Command::Set {
            key: vec![1],
            value: vec![10],
        });

        let result = state_machine.apply(&Command::Delete { key: vec![1] });

        assert_eq!(
            result,
            ApplyResult::Delete {
                previous: Some(vec![10])
            }
        );
        assert_eq!(state_machine.get(&[1]), None);
    }

    #[test]
    fn delete_of_an_absent_key_returns_none() {
        let mut state_machine = StateMachine::new();

        let result = state_machine.apply(&Command::Delete { key: vec![9] });

        assert_eq!(result, ApplyResult::Delete { previous: None });
    }

    #[test]
    fn get_returns_the_current_value_without_mutating_it() {
        let mut state_machine = StateMachine::new();
        state_machine.apply(&Command::Set {
            key: vec![1],
            value: vec![10],
        });

        let result = state_machine.apply(&Command::Get { key: vec![1] });

        assert_eq!(
            result,
            ApplyResult::Get {
                value: Some(vec![10])
            }
        );
        assert_eq!(state_machine.get(&[1]), Some(&[10][..]));
    }

    #[test]
    fn get_of_an_absent_key_returns_none() {
        let mut state_machine = StateMachine::new();

        let result = state_machine.apply(&Command::Get { key: vec![1] });

        assert_eq!(result, ApplyResult::Get { value: None });
    }

    #[test]
    fn apply_advances_last_applied_by_one_each_time() {
        let mut state_machine = StateMachine::new();
        assert_eq!(state_machine.last_applied(), 0);

        state_machine.apply(&Command::Get { key: vec![1] });
        assert_eq!(state_machine.last_applied(), 1);

        state_machine.apply(&Command::Get { key: vec![1] });
        assert_eq!(state_machine.last_applied(), 2);
    }

    #[test]
    fn propose_returns_none_when_not_leader() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let mut replica = Replica::new(node);

        let result = replica.propose(Command::Set {
            key: vec![1],
            value: vec![2],
        });

        assert_eq!(result, None);
        assert_eq!(replica.state_machine().last_applied(), 0);
    }

    #[test]
    fn propose_on_a_single_node_cluster_commits_and_applies_immediately() {
        let node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        let mut replica = Replica::new(node);
        replica.step(Event::Tick { next_timeout: 5 });
        assert_eq!(replica.node().role(), Role::Leader);

        let index = replica
            .propose(Command::Set {
                key: vec![1],
                value: vec![9],
            })
            .expect("leader accepts propose");

        assert_eq!(
            replica.applied_result(index),
            Some(&ApplyResult::Set { previous: None })
        );
        assert_eq!(replica.state_machine().get(&[1]), Some(&[9][..]));
    }

    #[test]
    fn repeated_steps_with_no_new_commits_never_reapply() {
        let node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        let mut replica = Replica::new(node);
        replica.step(Event::Tick { next_timeout: 5 });
        replica.propose(Command::Set {
            key: vec![1],
            value: vec![9],
        });
        assert_eq!(replica.state_machine().last_applied(), 1);

        replica.step(Event::Tick { next_timeout: 5 });
        replica.step(Event::Tick { next_timeout: 5 });

        assert_eq!(replica.state_machine().last_applied(), 1);
        assert_eq!(
            replica.applied_result(1),
            Some(&ApplyResult::Set { previous: None })
        );
    }

    fn three_replica_cluster(
        election_timeout: u64,
        heartbeat_interval: u64,
    ) -> BTreeMap<NodeId, Replica> {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        peers
            .iter()
            .map(|&id| {
                let node = Node::new(id, peers.clone(), election_timeout, heartbeat_interval)
                    .expect("valid node");
                (id, Replica::new(node))
            })
            .collect()
    }

    /// Delivers `effects` (sent by `from`) to their targets, then
    /// recursively delivers whatever those deliveries produce, until
    /// nothing remains in flight. Mirrors `kurogane-sim`'s
    /// `deliver_until_quiescent`, scoped to `Replica` instead of `Node`.
    fn deliver_until_quiescent(
        replicas: &mut BTreeMap<NodeId, Replica>,
        isolated: &[NodeId],
        from: NodeId,
        effects: Vec<Effect>,
    ) {
        fn sends(
            from: NodeId,
            effects: Vec<Effect>,
            isolated: &[NodeId],
        ) -> impl Iterator<Item = (NodeId, NodeId, Message)> + '_ {
            effects.into_iter().filter_map(move |effect| match effect {
                Effect::Send { to, message } if !isolated.contains(&to) => {
                    Some((from, to, message))
                }
                _ => None,
            })
        }

        let mut pending: Vec<(NodeId, NodeId, Message)> = sends(from, effects, isolated).collect();

        let mut guard = 0;
        while let Some((from, to, message)) = pending.pop() {
            guard += 1;
            assert!(guard < 10_000, "deliver_until_quiescent did not converge");

            let response = replicas
                .get_mut(&to)
                .expect("known replica")
                .step(Event::Step { from, message });
            pending.extend(sends(to, response, isolated));
        }
    }

    #[test]
    fn three_replicas_converge_after_a_mix_of_set_and_delete_commands() {
        let mut replicas = three_replica_cluster(1, 1);

        let requests = replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(1), requests);
        assert_eq!(replicas[&NodeId(1)].node().role(), Role::Leader);

        let leader = replicas.get_mut(&NodeId(1)).expect("known replica");
        leader.propose(Command::Set {
            key: vec![1],
            value: vec![10],
        });
        leader.propose(Command::Set {
            key: vec![2],
            value: vec![20],
        });
        leader.propose(Command::Delete { key: vec![1] });

        // First round replicates the entries and lets the leader see a
        // majority ack, committing locally. Followers only learn the leader
        // committed via leader_commit on a *later* AppendEntries, so a
        // second round is required before their own state machines catch up.
        for _ in 0..2 {
            let heartbeat = replicas
                .get_mut(&NodeId(1))
                .expect("known replica")
                .step(Event::Tick { next_timeout: 10 });
            deliver_until_quiescent(&mut replicas, &[], NodeId(1), heartbeat);
        }

        for id in [NodeId(1), NodeId(2), NodeId(3)] {
            let state = replicas[&id].state_machine();
            assert_eq!(state.get(&[1]), None);
            assert_eq!(state.get(&[2]), Some(&[20][..]));
        }
    }

    #[test]
    fn healed_partition_preserves_committed_commands_and_discards_uncommitted_ones() {
        let mut replicas = three_replica_cluster(1, 1);

        // Node 1 wins term 1 uncontested and commits a Set.
        let requests = replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(1), requests);
        assert_eq!(replicas[&NodeId(1)].node().role(), Role::Leader);

        replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .propose(Command::Set {
                key: b"x".to_vec(),
                value: b"committed".to_vec(),
            });
        let heartbeat = replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(1), heartbeat);
        assert_eq!(replicas[&NodeId(1)].node().commit_index(), 1);

        // Node 1 proposes one more command that never leaves its own log,
        // then is isolated before it can replicate.
        replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .propose(Command::Set {
                key: b"x".to_vec(),
                value: b"lost".to_vec(),
            });

        // Node 2 wins term 2 with node 3's vote; node 1 never hears about it.
        let node2_requests = replicas
            .get_mut(&NodeId(2))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[NodeId(1)], NodeId(2), node2_requests);
        assert_eq!(replicas[&NodeId(2)].node().role(), Role::Leader);

        replicas
            .get_mut(&NodeId(2))
            .expect("known replica")
            .propose(Command::Set {
                key: b"x".to_vec(),
                value: b"winner".to_vec(),
            });
        let node2_heartbeat = replicas
            .get_mut(&NodeId(2))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[NodeId(1)], NodeId(2), node2_heartbeat);
        assert_eq!(replicas[&NodeId(2)].node().commit_index(), 2);

        // The partition heals: node 1 hears from the new leader and
        // converges onto its committed history.
        let healing = replicas
            .get_mut(&NodeId(2))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(2), healing);

        for id in [NodeId(1), NodeId(2), NodeId(3)] {
            assert_eq!(
                replicas[&id].state_machine().get(b"x"),
                Some(&b"winner"[..])
            );
        }
    }

    #[test]
    fn a_get_routed_through_the_log_returns_the_previously_written_value() {
        let mut replicas = three_replica_cluster(1, 1);

        let requests = replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(1), requests);
        assert_eq!(replicas[&NodeId(1)].node().role(), Role::Leader);

        let leader = replicas.get_mut(&NodeId(1)).expect("known replica");
        leader.propose(Command::Set {
            key: vec![1],
            value: vec![42],
        });
        let get_index = leader
            .propose(Command::Get { key: vec![1] })
            .expect("leader accepts propose");

        let heartbeat = replicas
            .get_mut(&NodeId(1))
            .expect("known replica")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut replicas, &[], NodeId(1), heartbeat);

        assert_eq!(
            replicas[&NodeId(1)].applied_result(get_index),
            Some(&ApplyResult::Get {
                value: Some(vec![42])
            })
        );
    }
}
