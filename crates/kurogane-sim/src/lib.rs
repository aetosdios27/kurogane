//! Deterministic ownership of Raft nodes for controlled simulations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use kurogane_raft::{
    Effect, Event, HardState, LogEntry, Message, Node, NodeId, Role, SnapshotMetadata,
};

/// Invalid construction of a deterministic cluster.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterError {
    EmptyCluster,
    DuplicateNode(NodeId),
    MembershipMismatch(NodeId),
    MissingNode(NodeId),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCluster => formatter.write_str("cluster must contain at least one node"),
            Self::DuplicateNode(node) => {
                write!(formatter, "cluster contains duplicate node {node:?}")
            }
            Self::MembershipMismatch(node) => {
                write!(
                    formatter,
                    "node {node:?} has a different cluster membership"
                )
            }
            Self::MissingNode(node) => {
                write!(
                    formatter,
                    "cluster membership references missing node {node:?}"
                )
            }
        }
    }
}

impl Error for ClusterError {}

/// A fixed cluster whose node iteration order is canonical by [`NodeId`].
#[derive(Debug)]
pub struct Cluster {
    nodes: BTreeMap<NodeId, Node>,
}

impl Cluster {
    /// Takes ownership of one node for every member in a shared configuration.
    pub fn new(nodes: Vec<Node>) -> Result<Self, ClusterError> {
        if nodes.is_empty() {
            return Err(ClusterError::EmptyCluster);
        }

        let mut by_id = BTreeMap::new();
        for node in nodes {
            let id = node.id();
            if by_id.insert(id, node).is_some() {
                return Err(ClusterError::DuplicateNode(id));
            }
        }

        let expected_members = by_id
            .first_key_value()
            .expect("non-empty cluster checked above")
            .1
            .voters()
            .to_vec();

        for node in by_id.values() {
            if node.voters() != expected_members {
                return Err(ClusterError::MembershipMismatch(node.id()));
            }
        }
        for member in expected_members {
            if !by_id.contains_key(&member) {
                return Err(ClusterError::MissingNode(member));
            }
        }

        Ok(Self { nodes: by_id })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn node_ids(&self) -> impl ExactSizeIterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// Replaces one member's node in place — e.g. after simulating a crash
    /// and reconstructing it with `Node::recover`. The ID must already be a
    /// cluster member; this changes what a node *is*, not the membership.
    pub fn replace_node(&mut self, node: Node) {
        debug_assert!(
            self.nodes.contains_key(&node.id()),
            "replace_node must target an existing member"
        );
        self.nodes.insert(node.id(), node);
    }
}

/// One node's simulated durable storage: whatever `HardState`/log/snapshot
/// data have actually been confirmed via `Persist*` effects. A crash
/// discards everything else — reconstructing a node from a `DurableState` is
/// exactly what a real restart-from-disk would see.
#[derive(Clone, Debug, Default)]
pub struct DurableState {
    hard_state: HardState,
    log: Vec<LogEntry>,
    snapshot: SnapshotMetadata,
    snapshot_data: Vec<u8>,
}

impl DurableState {
    pub fn hard_state(&self) -> HardState {
        self.hard_state
    }

    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    pub fn snapshot(&self) -> SnapshotMetadata {
        self.snapshot
    }

    pub fn snapshot_data(&self) -> &[u8] {
        &self.snapshot_data
    }

    /// Records one effect as durably written. `Send` is not persistence and
    /// is ignored.
    pub fn apply(&mut self, effect: &Effect) {
        match effect {
            Effect::PersistHardState { term, voted_for } => {
                self.hard_state = HardState {
                    current_term: *term,
                    voted_for: *voted_for,
                };
            }
            Effect::PersistLog {
                from_index,
                entries,
            } => {
                // `from_index` is an absolute log index; `log[0]` holds
                // whatever comes right after the current snapshot boundary,
                // not necessarily absolute index 1.
                self.log
                    .truncate((*from_index - self.snapshot.last_included_index - 1) as usize);
                self.log.extend(entries.iter().cloned());
            }
            Effect::PersistSnapshot {
                last_included_index,
                last_included_term,
                data,
                // Real snapshot-config persistence in DurableState is
                // separate, later cross-crate work, not this stage's job --
                // ignored here only to keep the workspace compiling against
                // the widened Effect variant, same treatment as
                // PersistLearners below.
                config: _,
            } => {
                self.snapshot = SnapshotMetadata {
                    last_included_index: *last_included_index,
                    last_included_term: *last_included_term,
                };
                self.snapshot_data = data.clone();
            }
            Effect::Send { .. } => {}
            // Real learner-set persistence in DurableState is separate,
            // later cross-crate work, not this stage's job -- this arm
            // exists only to keep the workspace compiling against the new
            // Effect variant.
            Effect::PersistLearners { .. } => {}
        }
    }
}

/// Deterministic splitmix64 generator so simulations are reproducible from a seed.
///
/// Lives here, not in `kurogane-raft`: the core never reads time or randomness
/// itself, it only receives an explicit `next_timeout` on each `Event::Tick`.
#[derive(Debug)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in `[low, high]` inclusive.
    fn range_inclusive(&mut self, low: u64, high: u64) -> u64 {
        low + self.next_u64() % (high - low + 1)
    }
}

/// One recorded step of a simulation, for equality-based reproducibility checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TraceEvent {
    Tick {
        at: u64,
        node: NodeId,
        next_timeout: u64,
    },
    Delivered {
        at: u64,
        from: NodeId,
        to: NodeId,
        message: Message,
    },
    Sent {
        at: u64,
        from: NodeId,
        to: NodeId,
        message: Message,
    },
}

/// Drives a [`Cluster`] through logical time with randomized-but-seeded election
/// timeouts and message delivery delays, recording a full trace for replay checks.
pub struct Simulation {
    cluster: Cluster,
    rng: Rng,
    clock: u64,
    inbox: BTreeMap<u64, Vec<(NodeId, NodeId, Message)>>,
    min_timeout: u64,
    max_timeout: u64,
    min_delay: u64,
    max_delay: u64,
    trace: Vec<TraceEvent>,
}

impl Simulation {
    pub fn new(
        cluster: Cluster,
        seed: u64,
        min_timeout: u64,
        max_timeout: u64,
        min_delay: u64,
        max_delay: u64,
    ) -> Self {
        Self {
            cluster,
            rng: Rng::new(seed),
            clock: 0,
            inbox: BTreeMap::new(),
            min_timeout,
            max_timeout,
            min_delay,
            max_delay,
            trace: Vec::new(),
        }
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    /// Nodes currently in `Role::Leader`, paired with their current term.
    pub fn leaders(&self) -> Vec<(NodeId, u64)> {
        self.cluster
            .node_ids()
            .filter_map(|id| {
                let node = self.cluster.node(id).expect("known node id");
                (node.role() == Role::Leader).then(|| (id, node.current_term()))
            })
            .collect()
    }

    /// Advances logical time by one tick: delivers due messages, then ticks every
    /// node in canonical order.
    pub fn step(&mut self) {
        self.clock += 1;

        if let Some(due) = self.inbox.remove(&self.clock) {
            for (from, to, message) in due {
                self.trace.push(TraceEvent::Delivered {
                    at: self.clock,
                    from,
                    to,
                    message: message.clone(),
                });
                let effects = self
                    .cluster
                    .node_mut(to)
                    .expect("delivery target is a cluster member")
                    .step(Event::Step { from, message });
                self.apply_effects(to, effects);
            }
        }

        let ids: Vec<NodeId> = self.cluster.node_ids().collect();
        for id in ids {
            let next_timeout = self.rng.range_inclusive(self.min_timeout, self.max_timeout);
            self.trace.push(TraceEvent::Tick {
                at: self.clock,
                node: id,
                next_timeout,
            });
            let effects = self
                .cluster
                .node_mut(id)
                .expect("known node id")
                .step(Event::Tick { next_timeout });
            self.apply_effects(id, effects);
        }
    }

    fn apply_effects(&mut self, from: NodeId, effects: Vec<Effect>) {
        for effect in effects {
            let Effect::Send { to, message } = effect else {
                // Persist* effects aren't message delivery; nothing to route.
                continue;
            };
            self.trace.push(TraceEvent::Sent {
                at: self.clock,
                from,
                to,
                message: message.clone(),
            });
            let delay = self.rng.range_inclusive(self.min_delay, self.max_delay);
            self.inbox
                .entry(self.clock + delay)
                .or_default()
                .push((from, to, message));
        }
    }
}

#[cfg(test)]
mod tests {
    use kurogane_raft::{
        ClusterConfig, Effect, HardState, LogEntry, LogPayload, Node, NodeId, Snapshot,
        SnapshotMetadata,
    };

    use super::{Cluster, ClusterError, DurableState};

    fn node(id: u64, peers: &[NodeId]) -> Node {
        Node::new(NodeId(id), peers.to_vec(), 10 + id, 1).expect("valid node")
    }

    #[test]
    fn iterates_nodes_in_canonical_order() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let cluster = Cluster::new(vec![node(3, &peers), node(1, &peers), node(2, &peers)])
            .expect("valid cluster");

        assert_eq!(cluster.len(), 3);
        assert_eq!(
            cluster.node_ids().collect::<Vec<_>>(),
            vec![NodeId(1), NodeId(2), NodeId(3)]
        );
    }

    #[test]
    fn rejects_duplicate_nodes() {
        let peers = [NodeId(1)];
        let result = Cluster::new(vec![node(1, &peers), node(1, &peers)]);

        assert_eq!(
            result.expect_err("duplicate node must fail"),
            ClusterError::DuplicateNode(NodeId(1))
        );
    }

    #[test]
    fn rejects_inconsistent_or_incomplete_membership() {
        let shared = [NodeId(1), NodeId(2)];
        let different = [NodeId(1), NodeId(2), NodeId(3)];
        let mismatch = Cluster::new(vec![
            node(1, &shared),
            node(2, &different),
            node(3, &different),
        ]);
        assert_eq!(
            mismatch.expect_err("memberships must match"),
            ClusterError::MembershipMismatch(NodeId(2))
        );

        let incomplete = Cluster::new(vec![node(1, &shared)]);
        assert_eq!(
            incomplete.expect_err("all members must be present"),
            ClusterError::MissingNode(NodeId(2))
        );
    }

    #[test]
    fn replace_node_swaps_a_member_in_place() {
        let peers = [NodeId(1), NodeId(2)];
        let mut cluster =
            Cluster::new(vec![node(1, &peers), node(2, &peers)]).expect("valid cluster");

        let recovered = Node::recover(
            NodeId(1),
            peers.to_vec(),
            11,
            1,
            HardState {
                current_term: 7,
                voted_for: Some(NodeId(2)),
            },
            Vec::new(),
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");
        cluster.replace_node(recovered);

        assert_eq!(cluster.len(), 2);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            7
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").voted_for(),
            Some(NodeId(2))
        );
    }

    #[test]
    fn durable_state_accumulates_hard_state_and_splices_the_log() {
        let mut durable = DurableState::default();

        durable.apply(&Effect::PersistHardState {
            term: 1,
            voted_for: Some(NodeId(2)),
        });
        durable.apply(&Effect::PersistLog {
            from_index: 1,
            entries: vec![
                LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![1]),
                },
                LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![2]),
                },
            ],
        });

        assert_eq!(
            durable.hard_state(),
            HardState {
                current_term: 1,
                voted_for: Some(NodeId(2)),
            }
        );
        assert_eq!(durable.log().len(), 2);

        // A conflict-truncate at index 2 replaces the tail, same as
        // on_append_entries does in-memory.
        let replacement = LogEntry {
            term: 2,
            payload: LogPayload::Command(vec![9]),
        };
        durable.apply(&Effect::PersistHardState {
            term: 2,
            voted_for: None,
        });
        durable.apply(&Effect::PersistLog {
            from_index: 2,
            entries: vec![replacement.clone()],
        });

        assert_eq!(
            durable.hard_state(),
            HardState {
                current_term: 2,
                voted_for: None,
            }
        );
        assert_eq!(durable.log().len(), 2);
        assert_eq!(durable.log()[1], replacement);
    }

    #[test]
    fn durable_state_persist_snapshot_then_persist_log_uses_the_new_boundary() {
        let mut durable = DurableState::default();
        durable.apply(&Effect::PersistLog {
            from_index: 1,
            entries: vec![
                LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![1]),
                },
                LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![2]),
                },
                LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![3]),
                },
            ],
        });

        // Compacting through index 3, mirroring exactly what Node::compact
        // emits: PersistSnapshot moves the boundary, then a PersistLog
        // pins down what's retained above it -- here, nothing.
        durable.apply(&Effect::PersistSnapshot {
            last_included_index: 3,
            last_included_term: 1,
            data: vec![9, 9],
            config: ClusterConfig {
                voters: vec![NodeId(1)],
                old_voters: None,
            },
        });
        durable.apply(&Effect::PersistLog {
            from_index: 4,
            entries: Vec::new(),
        });

        // A later entry lands at absolute index 4 -- the first index above
        // the new boundary, not vec position 4 counted from the old start.
        durable.apply(&Effect::PersistLog {
            from_index: 4,
            entries: vec![LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![4]),
            }],
        });

        assert_eq!(
            durable.snapshot(),
            SnapshotMetadata {
                last_included_index: 3,
                last_included_term: 1,
            }
        );
        assert_eq!(durable.snapshot_data(), &[9, 9]);
        assert_eq!(
            durable.log(),
            &[LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![4])
            }]
        );
    }

    #[test]
    fn durable_state_ignores_send_effects() {
        use kurogane_raft::{Message, RequestVoteResponse};

        let mut durable = DurableState::default();

        durable.apply(&Effect::Send {
            to: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });

        assert_eq!(durable.hard_state(), HardState::default());
        assert!(durable.log().is_empty());
    }
}

#[cfg(test)]
mod simulation_tests {
    use std::collections::BTreeMap;

    use kurogane_raft::{
        AppendEntries, AppendEntriesResponse, ClusterConfig, Effect, Event, InstallSnapshot,
        InstallSnapshotResponse, LogEntry, LogPayload, Message, Node, NodeId, RequestVote,
        RequestVoteResponse, Role, Snapshot, SnapshotMetadata,
    };

    use super::{Cluster, DurableState, Simulation};

    fn three_node_cluster(election_timeout: u64, heartbeat_interval: u64) -> Cluster {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        Cluster::new(
            peers
                .iter()
                .map(|&id| {
                    Node::new(id, peers.to_vec(), election_timeout, heartbeat_interval)
                        .expect("valid node")
                })
                .collect(),
        )
        .expect("valid cluster")
    }

    /// Drives the simulation until a leader is observed, asserting along the way
    /// that at most one leader exists at a time and that a term never has two
    /// different leaders.
    fn run_until_leader_checking_invariants(simulation: &mut Simulation, max_ticks: u64) -> NodeId {
        let mut leader_by_term: BTreeMap<u64, NodeId> = BTreeMap::new();

        for _ in 0..max_ticks {
            simulation.step();

            let leaders = simulation.leaders();
            assert!(
                leaders.len() <= 1,
                "no more than one leader may exist at once, saw {leaders:?}"
            );

            for &(id, term) in &leaders {
                match leader_by_term.get(&term) {
                    Some(&existing) if existing != id => {
                        panic!(
                            "two different leaders observed in term {term}: {existing:?} and {id:?}"
                        );
                    }
                    _ => {
                        leader_by_term.insert(term, id);
                    }
                }
            }

            if let Some(&(id, _)) = leaders.first() {
                return id;
            }
        }

        panic!("cluster failed to elect a leader within {max_ticks} ticks");
    }

    /// Delivers `effects` (sent by `from`) to their targets over an
    /// instantaneous, fully-connected network, then recursively delivers
    /// whatever those deliveries produce, until nothing remains in flight.
    /// Node IDs in `isolated` never receive anything, simulating a partition.
    /// For hand-driven scenarios that don't need `Simulation`'s timing model.
    fn deliver_until_quiescent(
        cluster: &mut Cluster,
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

            let response = cluster
                .node_mut(to)
                .expect("known node")
                .step(Event::Step { from, message });
            pending.extend(sends(to, response, isolated));
        }
    }

    #[test]
    fn elects_a_single_leader_and_never_two_in_the_same_term() {
        let mut simulation = Simulation::new(three_node_cluster(3, 1), 42, 3, 6, 1, 2);

        run_until_leader_checking_invariants(&mut simulation, 200);
    }

    #[test]
    fn same_seed_and_schedule_reproduces_an_identical_trace() {
        let mut first = Simulation::new(three_node_cluster(3, 1), 7, 3, 6, 1, 2);
        let mut second = Simulation::new(three_node_cluster(3, 1), 7, 3, 6, 1, 2);

        for _ in 0..50 {
            first.step();
            second.step();
        }

        assert_eq!(first.trace(), second.trace());
    }

    #[test]
    fn stable_leadership_prevents_unnecessary_elections_under_delivered_heartbeats() {
        let mut simulation = Simulation::new(three_node_cluster(4, 1), 99, 4, 6, 1, 1);

        let leader = run_until_leader_checking_invariants(&mut simulation, 200);
        let leader_term = simulation
            .leaders()
            .into_iter()
            .find(|&(id, _)| id == leader)
            .map(|(_, term)| term)
            .expect("elected leader must report a term");

        // A short, fixed heartbeat interval and delivery delay well under the
        // election timeout should keep every follower's timer reset, so no
        // one else should ever become a candidate.
        for _ in 0..100 {
            simulation.step();
            assert_eq!(simulation.leaders(), vec![(leader, leader_term)]);
        }
    }

    #[test]
    fn an_isolated_old_leader_steps_down_once_a_higher_term_message_reaches_it() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        // Node 1 wins term 1 uncontested.
        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        for effect in requests {
            let Effect::Send { to, message } = effect else {
                continue;
            };
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for effect in responses {
                let Effect::Send {
                    to: respond_to,
                    message,
                } = effect
                else {
                    continue;
                };
                cluster
                    .node_mut(respond_to)
                    .expect("known node")
                    .step(Event::Step { from: to, message });
            }
        }
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            1
        );

        // Node 1 is isolated from here on: nothing is ever delivered to or from
        // it again until the healing step below. Nodes 2 and 3 time out on
        // their own and elect node 2 for term 2 without node 1 seeing anything.
        let node2_requests = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        for effect in node2_requests {
            let Effect::Send { to, message } = effect else {
                continue;
            };
            if to != NodeId(3) {
                continue;
            }
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(2),
                message,
            });
            for effect in responses {
                let Effect::Send {
                    to: respond_to,
                    message,
                } = effect
                else {
                    continue;
                };
                cluster
                    .node_mut(respond_to)
                    .expect("known node")
                    .step(Event::Step { from: to, message });
            }
        }
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").current_term(),
            2
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            1
        );

        // The partition heals: node 1 finally receives a heartbeat from the new
        // leader and steps down.
        let effects = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: Message::AppendEntries(AppendEntries {
                    term: 2,
                    leader_id: NodeId(2),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                }),
            });

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Follower
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            2
        );
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 2,
                        success: true,
                        match_index: 0,
                    }),
                }
            ]
        );
    }

    #[test]
    fn recovers_from_a_simultaneous_split_vote_via_a_later_timeout() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        // All three nodes time out in the same instant and become term-1 candidates
        // before any message is delivered.
        let mut requests = Vec::new();
        for &id in &peers {
            let effects = cluster
                .node_mut(id)
                .expect("known node")
                .step(Event::Tick { next_timeout: 2 });
            requests.extend(effects.into_iter().filter_map(|effect| match effect {
                Effect::Send { to, message } => Some((id, to, message)),
                _ => None,
            }));
        }
        for &id in &peers {
            let node = cluster.node(id).expect("known node");
            assert_eq!(node.role(), Role::Candidate);
            assert_eq!(node.current_term(), 1);
        }

        // Every node has already voted for itself, so nobody picks up an external
        // vote: a true split vote.
        for (from, to, message) in requests {
            let responses = cluster
                .node_mut(to)
                .expect("known node")
                .step(Event::Step { from, message });
            for effect in responses {
                let Effect::Send {
                    to: respond_to,
                    message,
                } = effect
                else {
                    continue;
                };
                cluster
                    .node_mut(respond_to)
                    .expect("known node")
                    .step(Event::Step { from: to, message });
            }
        }
        for &id in &peers {
            let node = cluster.node(id).expect("known node");
            assert_eq!(node.role(), Role::Candidate);
            assert_eq!(node.votes_granted().len(), 1);
        }

        // A later timeout lets node 1 alone retry for term 2; the others were never
        // ticked again, mirroring a longer randomized timeout on their side.
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 2 });
        let retry_effects = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 2 });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            2
        );

        for effect in retry_effects {
            let Effect::Send { to, message } = effect else {
                continue;
            };
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for effect in responses {
                let Effect::Send {
                    to: respond_to,
                    message,
                } = effect
                else {
                    continue;
                };
                cluster
                    .node_mut(respond_to)
                    .expect("known node")
                    .step(Event::Step { from: to, message });
            }
        }

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );
        for &id in &[NodeId(2), NodeId(3)] {
            let node = cluster.node(id).expect("known node");
            assert_eq!(node.role(), Role::Follower);
            assert_eq!(node.current_term(), 2);
        }
    }

    #[test]
    fn rejects_a_stale_term_request_vote_once_a_leader_is_established() {
        let mut cluster = three_node_cluster(1, 1);

        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        for effect in requests {
            let Effect::Send { to, message } = effect else {
                continue;
            };
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for effect in responses {
                let Effect::Send {
                    to: respond_to,
                    message,
                } = effect
                else {
                    continue;
                };
                cluster
                    .node_mut(respond_to)
                    .expect("known node")
                    .step(Event::Step { from: to, message });
            }
        }
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            1
        );

        let stale_request = RequestVote {
            term: 0,
            candidate_id: NodeId(2),
            last_log_index: 0,
            last_log_term: 0,
        };
        let effects = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: Message::RequestVote(stale_request),
            });

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            1
        );
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: false,
                }),
            }]
        );
    }

    #[test]
    fn a_follower_that_missed_early_replication_catches_up_on_the_next_heartbeat() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        for command in [vec![1u8], vec![2], vec![3]] {
            cluster
                .node_mut(NodeId(1))
                .expect("known node")
                .propose(command);
        }
        assert_eq!(
            cluster
                .node(NodeId(3))
                .expect("known node")
                .last_log_index(),
            0
        );

        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), heartbeat);

        let leaders_log = cluster.node(NodeId(1)).expect("known node").log().to_vec();
        assert_eq!(
            cluster.node(NodeId(3)).expect("known node").log(),
            leaders_log.as_slice()
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            3
        );
    }

    #[test]
    fn an_isolated_follower_catches_up_via_install_snapshot_once_the_leader_has_compacted_past_it()
    {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        // Node 1 wins term 1 uncontested; node 3 is isolated from this
        // point on, so it never sees any of what follows until healing.
        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[NodeId(3)], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        for command in [vec![1u8], vec![2], vec![3]] {
            cluster
                .node_mut(NodeId(1))
                .expect("known node")
                .propose(command);
        }
        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[NodeId(3)], NodeId(1), heartbeat);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            3
        );

        // The leader compacts its entire committed log into a snapshot --
        // node 3, still isolated, has none of this history.
        let leader = cluster.node_mut(NodeId(1)).expect("known node");
        leader.compact(3, vec![9, 9]).expect("3 is committed");
        assert!(leader.log().is_empty());

        // The partition heals: node 1's next heartbeat to node 3 must be
        // an InstallSnapshot, since node 3's next_index (still 1, its
        // initial seed from becoming leader) has fallen at or below the
        // leader's new boundary.
        let healing = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), healing);

        let node3 = cluster.node(NodeId(3)).expect("known node");
        assert_eq!(
            node3.snapshot(),
            SnapshotMetadata {
                last_included_index: 3,
                last_included_term: 1,
            }
        );
        assert_eq!(node3.snapshot_data(), &[9, 9]);
        assert_eq!(node3.commit_index(), 3);
        assert_eq!(node3.last_log_index(), 3);
        assert!(node3.log().is_empty());
    }

    #[test]
    fn a_delayed_or_duplicate_install_snapshot_does_not_regress_a_node_that_already_caught_up() {
        let mut cluster = three_node_cluster(1, 1);

        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        for command in [vec![1u8], vec![2]] {
            cluster
                .node_mut(NodeId(1))
                .expect("known node")
                .propose(command);
        }
        // First round replicates the entries and lets the leader see a
        // majority ack, committing locally. Followers only learn the
        // leader committed via leader_commit on a *later* AppendEntries,
        // so a second round is required before node 2 is genuinely caught
        // up (matches kurogane-kv's identical two-round pattern).
        for _ in 0..2 {
            let heartbeat = cluster
                .node_mut(NodeId(1))
                .expect("known node")
                .step(Event::Tick { next_timeout: 10 });
            deliver_until_quiescent(&mut cluster, &[], NodeId(1), heartbeat);
        }
        assert_eq!(cluster.node(NodeId(2)).expect("known node").log().len(), 2);
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").commit_index(),
            2
        );

        // A stray InstallSnapshot arrives at node 2 for a point it already
        // covers via ordinary replication (e.g. a retried/duplicated
        // message) -- it must be a no-op, not a regression.
        let node2_log_before = cluster.node(NodeId(2)).expect("known node").log().to_vec();
        let effects = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: Message::InstallSnapshot(InstallSnapshot {
                    term: 1,
                    leader_id: NodeId(1),
                    last_included_index: 1,
                    last_included_term: 1,
                    data: vec![0xFF],
                    config: ClusterConfig {
                        voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                        old_voters: None,
                    },
                }),
            });

        let node2 = cluster.node(NodeId(2)).expect("known node");
        assert_eq!(node2.log(), node2_log_before.as_slice());
        assert_eq!(node2.commit_index(), 2);
        assert_eq!(node2.snapshot(), SnapshotMetadata::default());
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(1),
                message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                    term: 1,
                    last_included_index: 1,
                }),
            }]
        );

        // The cluster still converges normally afterward -- the no-op
        // didn't leave anything in a broken state.
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose(vec![3]);
        let final_heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), final_heartbeat);

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            3
        );
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").log(),
            cluster.node(NodeId(1)).expect("known node").log()
        );
    }

    #[test]
    fn leader_replacement_preserves_committed_entries_and_converges_diverged_logs() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        // Node 1 wins term 1 uncontested and commits entry A on the full
        // cluster.
        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose(vec![b'A']);
        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), heartbeat);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            1
        );

        // Node 1 proposes one more entry that never leaves its own log, then
        // is isolated before it can replicate.
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose(vec![b'C']);
        assert_eq!(
            cluster
                .node(NodeId(1))
                .expect("known node")
                .last_log_index(),
            2
        );

        // Node 2 times out (node 1 never contacts it again) and wins term 2
        // with node 3's vote.
        let node2_requests = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[NodeId(1)], NodeId(2), node2_requests);
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").role(),
            Role::Leader
        );
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").current_term(),
            2
        );

        // Node 2 proposes and commits entry B on the majority that can hear
        // it.
        cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .propose(vec![b'B']);
        let node2_heartbeat = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[NodeId(1)], NodeId(2), node2_heartbeat);
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").commit_index(),
            2
        );
        assert_eq!(
            cluster
                .node(NodeId(3))
                .expect("known node")
                .last_log_index(),
            2
        );

        // The partition heals: node 1 hears from the new leader, steps down,
        // and its uncommitted entry C is discarded in favor of node 2's
        // committed B.
        let healing = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(2), healing);

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Follower
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            2
        );

        let committed_log = cluster.node(NodeId(2)).expect("known node").log().to_vec();
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").log(),
            committed_log.as_slice()
        );
        assert_eq!(
            cluster.node(NodeId(3)).expect("known node").log(),
            committed_log.as_slice()
        );
        assert!(
            committed_log.iter().all(|entry| !matches!(
                &entry.payload,
                LogPayload::Command(command) if command == &vec![b'C']
            )),
            "an uncommitted entry from a deposed leader must not survive"
        );
    }

    #[test]
    fn a_node_that_granted_a_vote_refuses_to_double_vote_after_a_simulated_crash() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers.clone(), 5, 2).expect("valid node");

        let mut durable = DurableState::default();
        for effect in &node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(2),
                last_log_index: 0,
                last_log_term: 0,
            }),
        }) {
            durable.apply(effect);
        }
        assert_eq!(node.voted_for(), Some(NodeId(2)));

        // The process crashes here: only what was actually persisted above
        // survives. Rebuilding from that (not from `node`, which is gone)
        // is what a real restart would see.
        let mut recovered = Node::recover(
            NodeId(1),
            peers,
            5,
            2,
            durable.hard_state(),
            durable.log().to_vec(),
            Snapshot {
                metadata: durable.snapshot(),
                data: durable.snapshot_data().to_vec(),
                config: ClusterConfig::default(),
            },
            Vec::new(),
        )
        .expect("valid node");
        assert_eq!(recovered.role(), Role::Follower);
        assert_eq!(recovered.current_term(), 1);
        assert_eq!(recovered.voted_for(), Some(NodeId(2)));

        let effects = recovered.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(recovered.voted_for(), Some(NodeId(2)));
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(3),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: false,
                }),
            }]
        );
    }

    #[test]
    fn a_followers_accepted_entries_survive_a_simulated_crash() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers.clone(), 5, 2).expect("valid node");

        let mut durable = DurableState::default();
        for effect in &node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    payload: LogPayload::Command(vec![7]),
                }],
                leader_commit: 0,
            }),
        }) {
            durable.apply(effect);
        }
        assert_eq!(node.log().len(), 1);

        let recovered = Node::recover(
            NodeId(1),
            peers,
            5,
            2,
            durable.hard_state(),
            durable.log().to_vec(),
            Snapshot {
                metadata: durable.snapshot(),
                data: durable.snapshot_data().to_vec(),
                config: ClusterConfig::default(),
            },
            Vec::new(),
        )
        .expect("valid node");

        assert_eq!(recovered.role(), Role::Follower);
        assert_eq!(recovered.current_term(), 1);
        assert_eq!(
            recovered.log(),
            &[LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![7]),
            }]
        );
    }

    #[test]
    fn a_leaders_proposed_entry_survives_a_simulated_crash_and_recovers_as_a_follower() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers.clone(), 1, 1).expect("valid node");

        let mut durable = DurableState::default();
        for effect in &node.step(Event::Tick { next_timeout: 5 }) {
            durable.apply(effect);
        }
        for effect in &node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        }) {
            durable.apply(effect);
        }
        assert_eq!(node.role(), Role::Leader);

        let (_, effects) = node.propose(vec![9]).expect("leader accepts propose");
        for effect in &effects {
            durable.apply(effect);
        }

        // The leader crashes here, having only ever announced its candidacy
        // and proposed one entry -- it never told anyone the entry existed.
        let recovered = Node::recover(
            NodeId(1),
            peers,
            1,
            1,
            durable.hard_state(),
            durable.log().to_vec(),
            Snapshot {
                metadata: durable.snapshot(),
                data: durable.snapshot_data().to_vec(),
                config: ClusterConfig::default(),
            },
            Vec::new(),
        )
        .expect("valid node");

        assert_eq!(recovered.role(), Role::Follower);
        assert_eq!(recovered.current_term(), 1);
        assert_eq!(
            recovered.log(),
            &[LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![9]),
            }]
        );
    }
}
