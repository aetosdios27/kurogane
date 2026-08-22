//! Deterministic ownership of Raft nodes for controlled simulations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use kurogane_raft::{
    ClusterConfig, Effect, Event, HardState, LogEntry, Message, Node, NodeId, Role,
    SnapshotMetadata,
};

/// Invalid construction of a deterministic cluster.
///
/// `MembershipMismatch`/`MissingNode` used to also live here, rejecting any
/// cluster whose members' voter lists weren't byte-for-byte identical. Joint
/// consensus makes that fundamentally the wrong check: nodes legitimately
/// hold divergent config views mid-transition (a lagging follower's log may
/// not yet have the joint entry the leader already applied), and `Cluster`
/// routing (`Simulation::step`/`apply_effects`) is a pure `NodeId` ->
/// `BTreeMap` lookup that never consults any node's config, so there was
/// never a safety reason to require agreement up front. Dropped outright
/// rather than narrowed, since every real invariant this was protecting
/// (unique IDs, non-empty cluster) is still covered by the two variants
/// below.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClusterError {
    EmptyCluster,
    DuplicateNode(NodeId),
}

impl fmt::Display for ClusterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCluster => formatter.write_str("cluster must contain at least one node"),
            Self::DuplicateNode(node) => {
                write!(formatter, "cluster contains duplicate node {node:?}")
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
    /// Takes ownership of one node per member. Members are not required to
    /// share an identical config view -- see `ClusterError`'s doc comment
    /// for why that's no longer enforced here.
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
    /// The membership active as of `snapshot`'s boundary, captured from
    /// `Effect::PersistSnapshot`'s own `config` field -- mirrors
    /// `Node::snapshot_config`, the field it's replayed into on recovery.
    snapshot_config: ClusterConfig,
    /// The learner set, captured from `Effect::PersistLearners` -- durably
    /// persisted independently of hard state/log/snapshot timing, same as
    /// `Node::recover`'s own `learners` parameter.
    learners: Vec<NodeId>,
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

    pub fn snapshot_config(&self) -> &ClusterConfig {
        &self.snapshot_config
    }

    pub fn learners(&self) -> &[NodeId] {
        &self.learners
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
                config,
            } => {
                self.snapshot = SnapshotMetadata {
                    last_included_index: *last_included_index,
                    last_included_term: *last_included_term,
                };
                self.snapshot_data = data.clone();
                self.snapshot_config = config.clone();
            }
            Effect::Send { .. } => {}
            Effect::PersistLearners { learners } => {
                self.learners = learners.clone();
            }
        }
    }
}

/// Deterministic splitmix64 generator so simulations are reproducible from a seed.
///
/// Lives here, not in `kurogane-raft`: the core never reads time or randomness
/// itself, it only receives an explicit `next_timeout` on each `Event::Tick`.
///
/// Public so a driver built on top of this crate's other pieces (e.g. one
/// that schedules its own client operations and fault injection over
/// `kurogane-kv::Replica`s rather than through `Cluster`/`Simulation`
/// directly) draws from the exact same reproducible stream `Simulation`
/// itself uses, instead of a second, unsynced generator.
#[derive(Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Returns a value in `[low, high]` inclusive.
    pub fn range_inclusive(&mut self, low: u64, high: u64) -> u64 {
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

    /// Reads one node's current state without mutating anything -- e.g. to
    /// discover the current leader's own `current_config`/`voters()` before
    /// deciding what config-change operation to drive next.
    pub fn node(&self, id: NodeId) -> Option<&Node> {
        self.cluster.node(id)
    }

    /// Drives `leader`'s `Node::add_learner(id)` and routes whatever
    /// effects it produces (just `Effect::PersistLearners`, no `Send`, per
    /// its own doc comment) through the same scheduler every other
    /// mutation goes through, so a config-change schedule driven through
    /// this method still replays identically under
    /// `same_seed_and_schedule_reproduces_an_identical_trace`-style replay.
    /// A no-op if `leader` isn't a known node.
    pub fn add_learner(&mut self, leader: NodeId, id: NodeId) {
        let Some(node) = self.cluster.node_mut(leader) else {
            return;
        };
        let effects = node.add_learner(id);
        self.apply_effects(leader, effects);
    }

    /// Drives `leader`'s `Node::propose_config_change(new_voters)` and
    /// routes whatever effects it produces through the scheduler, mirroring
    /// `add_learner` above. Returns the new entry's index, or `None` if
    /// `leader` isn't a known node or isn't actually the leader (or
    /// `new_voters` is empty) -- exactly `propose_config_change`'s own
    /// `None` cases.
    pub fn propose_config_change(
        &mut self,
        leader: NodeId,
        new_voters: Vec<NodeId>,
    ) -> Option<u64> {
        let node = self.cluster.node_mut(leader)?;
        let (index, effects) = node.propose_config_change(new_voters)?;
        self.apply_effects(leader, effects);
        Some(index)
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
    fn allows_divergent_or_incomplete_membership_views() {
        // This is a deliberate behavior change, not a regression: Cluster::new
        // used to reject any set of nodes whose voter lists weren't
        // byte-for-byte identical (ClusterError::MembershipMismatch/
        // MissingNode, both since removed). Joint consensus makes that check
        // actively wrong -- nodes legitimately hold divergent config views
        // mid-transition (a lagging follower's log may not yet have the
        // joint entry the leader already applied), and routing
        // (Simulation::step/apply_effects) is a pure NodeId -> BTreeMap
        // lookup that never consults any node's config, so there was never a
        // safety reason to require agreement up front.
        let shared = [NodeId(1), NodeId(2)];
        let different = [NodeId(1), NodeId(2), NodeId(3)];
        let divergent = Cluster::new(vec![
            node(1, &shared),
            node(2, &different),
            node(3, &different),
        ]);
        assert!(
            divergent.is_ok(),
            "divergent membership views must be accepted, not rejected"
        );

        let incomplete = Cluster::new(vec![node(1, &shared)]);
        assert!(
            incomplete.is_ok(),
            "a node whose own voter list names an absent member must be accepted -- \
             Cluster::new doesn't consult any node's config at all"
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
        assert_eq!(
            durable.snapshot_config(),
            &ClusterConfig {
                voters: vec![NodeId(1)],
                old_voters: None,
            }
        );
    }

    #[test]
    fn durable_state_persists_learners_and_a_recovered_node_prefers_the_snapshots_own_config() {
        let mut durable = DurableState::default();

        // A learner is tracked independently of any snapshot/log timing.
        durable.apply(&Effect::PersistLearners {
            learners: vec![NodeId(4)],
        });
        assert_eq!(durable.learners(), &[NodeId(4)]);

        // A later compaction captures the config active as of its boundary
        // -- here, a joint config mid-transition, to prove the full byte
        // (well, field) actually survives the round trip through
        // DurableState and back into a recovered Node, not just DurableState
        // itself. This is what makes gap 1's Node::recover fix meaningful:
        // without it, the recovered node below would end up with
        // current_config().voters == stale_peers instead.
        let real_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(2), NodeId(4)],
            old_voters: Some(vec![NodeId(1), NodeId(2)]),
        };
        durable.apply(&Effect::PersistSnapshot {
            last_included_index: 5,
            last_included_term: 1,
            data: vec![1, 2, 3],
            config: real_config.clone(),
        });
        assert_eq!(durable.snapshot_config(), &real_config);

        let stale_peers = vec![NodeId(1), NodeId(2)];
        let recovered = Node::recover(
            NodeId(1),
            stale_peers,
            1,
            1,
            durable.hard_state(),
            durable.log().to_vec(),
            Snapshot {
                metadata: durable.snapshot(),
                data: durable.snapshot_data().to_vec(),
                config: durable.snapshot_config().clone(),
            },
            durable.learners().to_vec(),
        )
        .expect("valid node");

        assert_eq!(recovered.current_config(), &real_config);
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
        AppendEntries, AppendEntriesResponse, ClusterConfig, Effect, Event, HardState,
        InstallSnapshot, InstallSnapshotResponse, LogEntry, LogPayload, Message, Node, NodeId,
        RequestVote, RequestVoteResponse, Role, Snapshot, SnapshotMetadata,
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

    /// The `Message` (if any) that `effects` sends to `to`. For hand-driven
    /// scenarios that need to selectively deliver only some of a batch of
    /// `Send`s (e.g. withholding one peer's copy of a heartbeat to control
    /// exactly which acks land first), where `deliver_until_quiescent`'s
    /// full recursive delivery would give away too much control.
    fn message_to(effects: &[Effect], to: NodeId) -> Option<Message> {
        let mut found = None;
        for effect in effects {
            if let Effect::Send {
                to: target,
                message,
            } = effect
            {
                if *target == to {
                    assert!(found.is_none(), "expected at most one message to {to:?}");
                    found = Some(message.clone());
                }
            }
        }
        found
    }

    /// Asserts no more than one node in `cluster` currently believes itself
    /// `Role::Leader` -- the `Cluster`-level counterpart to
    /// `run_until_leader_checking_invariants`'s per-tick check, for
    /// hand-driven scenarios that drive a `Cluster` directly instead of
    /// through `Simulation`.
    fn assert_at_most_one_leader(cluster: &Cluster) {
        let leaders: Vec<(NodeId, u64)> = cluster
            .node_ids()
            .filter_map(|id| {
                let node = cluster.node(id).expect("known node");
                (node.role() == Role::Leader).then(|| (id, node.current_term()))
            })
            .collect();
        assert!(
            leaders.len() <= 1,
            "no more than one leader may exist at once, saw {leaders:?}"
        );
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
    #[test]
    fn add_a_voter_to_a_three_node_cluster_requires_old_and_new_majority_through_real_delivery() {
        // Mirrors kurogane-raft's own
        // a_joint_commit_requires_match_progress_on_both_the_old_and_new_voters
        // test, but through kurogane-sim's Cluster/real message delivery
        // instead of direct Node calls. For a pure single-voter add from 3
        // to 4, "new-set-majority-alone" is mathematically unconstructible
        // as an insufficient case (every 3-of-4 combination necessarily
        // includes at least 2 of the original 3) -- see the smaller,
        // literal mirror below for that direction. What *is* constructible
        // here, and is the realistic danger the dual-majority rule guards
        // against, is the reverse: old-set majority reached without the
        // new voter having caught up at all.
        let mut cluster = Cluster::new(vec![
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            Node::new(NodeId(3), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            // Node::new_learner doesn't exist yet, so node 4 stands in for
            // a real joiner: seeded with the eventual full voter set up
            // front and a deliberately huge election timeout, so it can
            // never self-elect before being genuinely admitted -- it's
            // never ticked in this test at all.
            Node::new(
                NodeId(4),
                vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
                1_000_000,
                1,
            )
            .expect("valid node"),
        ])
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

        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .add_learner(NodeId(4));
        let (index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)])
            .expect("leader accepts config change");

        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });

        // Deliver only to node 2, deliberately withholding node 3 and node
        // 4.
        let node2_message = message_to(&heartbeat, NodeId(2)).expect("leader replicates to node 2");
        let node2_response = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: node2_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: message_to(&node2_response, NodeId(1)).expect("node 2 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            0,
            "old-set majority alone (self + node 2, of 3), before the new voter has caught \
             up at all, must not commit"
        );

        // Node 4 -- the new voter -- catches up too, completing the new
        // set's own majority (self + 2 + 4, of 4) now that old-set
        // majority was already satisfied.
        let node4_message = message_to(&heartbeat, NodeId(4)).expect("leader replicates to node 4");
        let node4_response = cluster
            .node_mut(NodeId(4))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: node4_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(4),
                message: message_to(&node4_response, NodeId(1)).expect("node 4 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            index
        );
    }

    #[test]
    fn a_follower_partitioned_across_a_membership_change_rejoins_once_healed_even_under_a_new_leader()
     {
        // Reproduces a suspected availability gap in is_member's local-config
        // gate, which has no analog anywhere in the paper's Figure 2: node 3
        // is partitioned for the entire membership change that adds node 4,
        // so node 3's own current_config never advances past the original
        // C_old = {1, 2, 3} -- it never sees the Configuration entries, only
        // the nodes that stayed reachable do. If the *next* leader elected
        // is drawn from the new membership (node 4) rather than a node 3
        // already recognizes, node 3's own is_member(4) check (still
        // evaluated against its own stale config) rejects every AppendEntries
        // node 4 ever sends it -- even after the partition heals -- because
        // step()'s dispatch drops the message before on_append_entries ever
        // runs. Node 3 can't win an election either (its log is behind), so
        // without a fix it is stuck outside the cluster forever, even though
        // node 4 is a completely legitimate, fully-caught-up leader.
        let mut cluster = Cluster::new(vec![
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            Node::new(NodeId(3), vec![NodeId(1), NodeId(2), NodeId(3)], 1, 1).expect("valid node"),
            Node::new(
                NodeId(4),
                vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
                1,
                1,
            )
            .expect("valid node"),
        ])
        .expect("valid cluster");

        // Node 1 wins the initial election with every real peer reachable --
        // node 3 is a completely ordinary member up to this point.
        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        // From here on, node 3 is partitioned -- every delivery in this test
        // withholds it. It will never see the upcoming membership change.
        let isolated = [NodeId(3)];

        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .add_learner(NodeId(4));
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)])
            .expect("leader accepts config change");

        // Drive enough heartbeat rounds for the joint entry to commit, the
        // automatic C_old,new -> C_new follow-up to be appended, and that
        // follow-up to commit too -- all via node 1 and node 2 alone (a real
        // majority of both the old {1,2,3} and new {1,2,3,4} sets without
        // node 3: node 1+2 is 2-of-3 old, node 1+2+4 is 3-of-4 new).
        for _ in 0..10 {
            let heartbeat = cluster
                .node_mut(NodeId(1))
                .expect("known node")
                .step(Event::Tick { next_timeout: 10 });
            deliver_until_quiescent(&mut cluster, &isolated, NodeId(1), heartbeat);
        }

        let expected_voters = vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)];
        for id in [NodeId(1), NodeId(2), NodeId(4)] {
            let node = cluster.node(id).expect("known node");
            assert_eq!(
                node.current_config().voters,
                expected_voters,
                "node {id:?} should have committed the plain C_new configuration"
            );
            assert!(
                node.current_config().old_voters.is_none(),
                "node {id:?} should be past the joint phase"
            );
        }
        // The whole point of this scenario: node 3, partitioned throughout,
        // never saw any of this.
        assert_eq!(
            cluster
                .node(NodeId(3))
                .expect("known node")
                .current_config()
                .voters,
            vec![NodeId(1), NodeId(2), NodeId(3)],
            "node 3 was partitioned for the entire membership change and must still be on C_old"
        );

        // Node 1 (the original leader) now crashes and restarts -- a
        // completely ordinary event, simulated the same way every other
        // crash-recovery test in this file does: reconstruct a fresh
        // Node::recover from its own persisted-equivalent state. It comes
        // back as a plain Follower, per Node::recover's own contract.
        let old_leader = cluster.node(NodeId(1)).expect("known node");
        let recovered = Node::recover(
            NodeId(1),
            vec![NodeId(1), NodeId(2), NodeId(3)],
            1,
            1,
            HardState {
                current_term: old_leader.current_term(),
                voted_for: None,
            },
            old_leader.log().to_vec(),
            Snapshot {
                metadata: old_leader.snapshot(),
                data: old_leader.snapshot_data().to_vec(),
                config: old_leader.snapshot_config().clone(),
            },
            old_leader.learners().iter().copied().collect(),
        )
        .expect("valid recovered node");
        cluster.replace_node(recovered);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Follower
        );

        // Node 4 campaigns. It's a real, fully-caught-up member of the
        // committed C_new, and it only asks the peers it (correctly)
        // recognizes -- 1, 2, and 3 -- but node 3 is still partitioned, so
        // only 1 and 2 actually receive it.
        let campaign = cluster
            .node_mut(NodeId(4))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &isolated, NodeId(4), campaign);
        assert_eq!(
            cluster.node(NodeId(4)).expect("known node").role(),
            Role::Leader,
            "node 4 should win with votes from 1, 2, and itself -- a real majority of C_new \
             that doesn't need node 3 at all"
        );

        // Heal the partition and let node 4, the new (and entirely
        // legitimate) leader, reach node 3 for the first time since it fell
        // behind.
        let heartbeat = cluster
            .node_mut(NodeId(4))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(4), heartbeat);

        assert_eq!(
            cluster
                .node(NodeId(3))
                .expect("known node")
                .current_config()
                .voters,
            expected_voters,
            "node 3 must catch up once the partition heals, even though the leader reaching it \
             (node 4) is not a member of node 3's own stale, pre-partition view of the cluster"
        );
    }

    #[test]
    fn add_a_voter_new_set_majority_alone_does_not_commit_without_the_old_set_too() {
        // The literal, discriminating mirror of kurogane-raft's own
        // a_joint_commit_requires_match_progress_on_both_the_old_and_new_voters
        // test (old {1, 4}, new {1, 4, 5} there): a 2-voter cluster adding
        // a 3rd is the smallest shape where new-set majority actually can
        // be reached without old-set majority also being satisfied -- a
        // 3-to-4 add (the test above) can't produce that combination at
        // all.
        let mut cluster = Cluster::new(vec![
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2)], 1, 1).expect("valid node"),
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2)], 1, 1).expect("valid node"),
            Node::new(
                NodeId(3),
                vec![NodeId(1), NodeId(2), NodeId(3)],
                1_000_000,
                1,
            )
            .expect("valid node"),
        ])
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

        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .add_learner(NodeId(3));
        let (index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3)])
            .expect("leader accepts config change");

        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });

        // Only node 3 -- the new voter -- catches up. self + 3 satisfies
        // the new set's majority (2 of 3), but the old set {1, 2} still
        // has only self's own progress (1 of 2), below its own quorum.
        let node3_message = message_to(&heartbeat, NodeId(3)).expect("leader replicates to node 3");
        let node3_response = cluster
            .node_mut(NodeId(3))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: node3_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(3),
                message: message_to(&node3_response, NodeId(1)).expect("node 3 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            0,
            "new-set majority alone, without old-set majority, must not commit"
        );

        // Node 2 -- the sole other old-set member -- now also acks,
        // completing old-set majority (self + 2, of 2) while new-set
        // majority was already satisfied.
        let node2_message = message_to(&heartbeat, NodeId(2)).expect("leader replicates to node 2");
        let node2_response = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: node2_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: message_to(&node2_response, NodeId(1)).expect("node 2 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            index
        );
    }

    #[test]
    fn remove_a_voter_including_leader_removal_steps_down_only_once_the_plain_config_commits() {
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
        assert_at_most_one_leader(&cluster);

        // The leader proposes removing itself: joint config, old {1,2,3},
        // new {2,3}.
        let (joint_index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(2), NodeId(3)])
            .expect("leader accepts config change");

        // Replicate and let the joint entry reach dual majority: old-set
        // majority (2 of 3: self + either) and new-set majority (both 2
        // and 3, since the new set has no overlap with the leader at all)
        // are both satisfied once nodes 2 and 3 ack.
        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), heartbeat);

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            joint_index,
            "the joint config must have committed"
        );
        // The leader appends the automatic C_new follow-up the instant the
        // joint entry commits, even before that entry itself commits --
        // current_config already excludes node 1 at this point, but
        // commit_index hasn't caught up to it yet, so node 1 must still be
        // leader here.
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader,
            "the leader must not step down on the joint config committing -- only on the \
             plain C_new"
        );
        assert_at_most_one_leader(&cluster);

        // A second replication round is needed to carry the C_new
        // follow-up the rest of the way (deliver_until_quiescent's
        // reactive cascade above only re-contacts a peer that just sent a
        // fresh ack -- node 3 never gets a second chance within that same
        // pass). Once both nodes 2 and 3 ack it, C_new's own commit fires
        // the leader's not-in-C_new step-down.
        let heartbeat2 = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), heartbeat2);

        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            joint_index + 1,
            "the plain C_new follow-up must have committed too"
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Follower,
            "the leader must step down once (and only once) C_new itself commits"
        );
        assert_at_most_one_leader(&cluster);
    }

    #[test]
    fn a_learners_match_index_never_affects_commit_index_until_it_is_promoted() {
        // A 2-voter base {1, 2} plus a learner (node 3), mirroring
        // add_a_voter_new_set_majority_alone_does_not_commit_without_the_old_set_too's
        // setup: it's the smallest shape where "count the learner in
        // quorum" and "don't" actually disagree. With a 3-voter base, self
        // + the learner alone would read as uncommitted either way (2 of a
        // hypothetical 4-member union is still below its quorum of 3) --
        // not a real test. Here, self + the learner alone is 2 of 2 under
        // a (hypothetical, buggy) 3-member union {1,2,3} (quorum 2) --
        // which would incorrectly commit -- versus the correct
        // computation, which only ever consults voters = {1, 2}: self
        // (caught up) and node 2 (still unacked, at 0) give candidate 0,
        // not committed. Every assertion below checks this against real
        // delivered acks, not by construction.
        let mut cluster = Cluster::new(vec![
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2)], 1, 1).expect("valid node"),
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2)], 1, 1).expect("valid node"),
            // Node::new_learner doesn't exist yet, so node 3 stands in for
            // a real joiner: seeded with the eventual full voter set up
            // front and a deliberately huge election timeout, so it can
            // never self-elect before being genuinely admitted below.
            Node::new(
                NodeId(3),
                vec![NodeId(1), NodeId(2), NodeId(3)],
                1_000_000,
                1,
            )
            .expect("valid node"),
        ])
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

        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .add_learner(NodeId(3));

        let (index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose(vec![9])
            .expect("leader accepts propose");

        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });

        // Deliver to the learner ONLY, through the same real
        // Cluster/message delivery mechanism a voter uses -- proving it
        // receives ordinary AppendEntries replication -- then feed its
        // (fully caught-up) match_index claim back to the leader, still
        // withholding node 2 entirely.
        let learner_message =
            message_to(&heartbeat, NodeId(3)).expect("the leader replicates to learners too");
        let learner_response = cluster
            .node_mut(NodeId(3))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: learner_message,
            });
        assert_eq!(
            cluster.node(NodeId(3)).expect("known node").log().len(),
            1,
            "the learner must actually receive and apply the entry"
        );
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(3),
                message: message_to(&learner_response, NodeId(1)).expect("the learner acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            0,
            "self + the learner alone must not commit -- advance_commit_index only ever \
             consults voters ({{1, 2}} here), never learners, so node 2's own progress (still \
             0) is what actually gates this, no matter how caught up the learner claims to be"
        );

        // Node 2 -- the sole other real voter -- now also acks, reaching
        // the voters-only quorum (2 of 2) on its own.
        let voter_message =
            message_to(&heartbeat, NodeId(2)).expect("the leader replicates to node 2");
        let voter_response = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: voter_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: message_to(&voter_response, NodeId(1)).expect("node 2 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            index
        );

        // Promote the learner to a full voter and drive the resulting
        // joint transition (old {1,2}, new {1,2,3}) to completion -- same
        // two-round shape as the remove-a-voter scenario, since the
        // automatic C_new follow-up needs its own replication round.
        let (promote_index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3)])
            .expect("leader accepts config change");
        let promote_round1 = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), promote_round1);
        let promote_round2 = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), promote_round2);

        assert!(
            cluster.node(NodeId(1)).expect("known node").commit_index() > promote_index,
            "the promotion's automatic C_new follow-up must have committed too"
        );
        assert!(
            cluster
                .node(NodeId(1))
                .expect("known node")
                .voters()
                .contains(&NodeId(3)),
            "node 3 must now be a real voter"
        );

        // Node 3 now genuinely counts toward quorum: propose one more
        // entry and deliver it to node 3 ONLY (withholding node 2 this
        // time) -- the exact same delivery shape as the pre-promotion
        // check above, but now committing, since self + node 3 (2 of the
        // now-3 voters) meets the quorum on its own.
        let (final_index, _effects) = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose(vec![10])
            .expect("leader accepts propose");
        let final_heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });

        let message = message_to(&final_heartbeat, NodeId(3)).expect("leader replicates to node 3");
        let response = cluster
            .node_mut(NodeId(3))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(3),
                message: message_to(&response, NodeId(1)).expect("node 3 acks"),
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").commit_index(),
            final_index,
            "self + node 3 alone must now be enough -- node 3 is a real voter, not a learner, \
             so its ack alone (without node 2) completes the quorum"
        );
    }

    #[test]
    fn a_removed_servers_stale_request_vote_does_not_depose_anyone_during_the_joint_window() {
        let peers = [NodeId(1), NodeId(2), NodeId(3)];
        let mut cluster = Cluster::new(
            peers
                .iter()
                .map(|&id| Node::new(id, peers.to_vec(), 1, 1).expect("valid node"))
                .collect(),
        )
        .expect("valid cluster");

        // Node 1 wins term 1 uncontested; nodes 2 and 3 both just heard
        // from it, so both have leader_contact_elapsed == 0.
        let requests = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 10 });
        deliver_until_quiescent(&mut cluster, &[], NodeId(1), requests);
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader
        );

        // The leader proposes removing node 3: joint config, old {1,2,3},
        // new {1,2}. This takes effect on the leader immediately, even
        // before it commits.
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .propose_config_change(vec![NodeId(1), NodeId(2)])
            .expect("leader accepts config change");
        assert!(
            cluster
                .node(NodeId(1))
                .expect("known node")
                .current_config()
                .is_joint(),
            "the joint config must be active on the leader immediately"
        );

        // Node 3 is partitioned from here on: nothing is delivered to or
        // from it through ordinary routing again, but it still exists in
        // `cluster` and keeps ticking on its own like a real isolated
        // process would, eventually timing out and starting a new
        // election at a higher term -- exactly the scenario the
        // disruption guard exists for.
        let mut node3_campaign = Vec::new();
        for _ in 0..5 {
            node3_campaign = cluster
                .node_mut(NodeId(3))
                .expect("known node")
                .step(Event::Tick { next_timeout: 5 });
            if !node3_campaign.is_empty() {
                break;
            }
        }
        let stale_term = cluster.node(NodeId(3)).expect("known node").current_term();
        assert!(
            stale_term > 1,
            "node 3 must have started a new, higher-term election"
        );

        // Deliver node 3's stale RequestVote straight to the leader. The
        // Role::Leader half of the guard is unconditional (not scoped to
        // the joint window), but the joint config is still active on the
        // leader right now too, since nothing has replicated yet.
        let leader_message =
            message_to(&node3_campaign, NodeId(1)).expect("node 3 requests node 1's vote");
        let leader_response = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(3),
                message: leader_message,
            });
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").role(),
            Role::Leader,
            "the leader must not step down on a RequestVote, even a higher-term one"
        );
        assert_eq!(
            cluster.node(NodeId(1)).expect("known node").current_term(),
            1
        );
        assert_eq!(
            leader_response,
            vec![Effect::Send {
                to: NodeId(3),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: false,
                }),
            }]
        );

        // Now replicate the joint entry to node 2 only (node 3 stays
        // isolated) and let the leader observe node 2's ack -- this
        // reaches dual majority (old {1,2,3}: self + 2; new {1,2}: self +
        // 2, both members) and commits the joint entry, which immediately
        // appends the automatic C_new follow-up on the leader. Node 2 has
        // NOT received that follow-up yet -- it needs a second heartbeat
        // round -- so node 2 is still in the joint window right now. This
        // is exactly the window this test needs to exercise: once node 2
        // also gets the follow-up, is_member alone would already drop
        // node 3's messages, which proves nothing about this guard.
        let heartbeat = cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Tick { next_timeout: 5 });
        let node2_message = message_to(&heartbeat, NodeId(2)).expect("leader replicates to node 2");
        let node2_response = cluster
            .node_mut(NodeId(2))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(1),
                message: node2_message,
            });
        cluster
            .node_mut(NodeId(1))
            .expect("known node")
            .step(Event::Step {
                from: NodeId(2),
                message: message_to(&node2_response, NodeId(1)).expect("node 2 acks"),
            });
        assert!(
            cluster
                .node(NodeId(2))
                .expect("known node")
                .current_config()
                .is_joint(),
            "node 2 must still be in the joint window -- it hasn't received the C_new \
             follow-up yet"
        );

        // Node 2 has not been ticked since the AppendEntries above reset
        // its leader_contact_elapsed to 0, so the joint-scoped
        // Follower/Candidate guard must fire: node 3's stale, higher-term
        // RequestVote must be rejected without granting or stepping down.
        let node2_message =
            message_to(&node3_campaign, NodeId(2)).expect("node 3 requests node 2's vote");
        let node2_guard_response =
            cluster
                .node_mut(NodeId(2))
                .expect("known node")
                .step(Event::Step {
                    from: NodeId(3),
                    message: node2_message,
                });
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").role(),
            Role::Follower
        );
        assert_eq!(
            cluster.node(NodeId(2)).expect("known node").current_term(),
            1
        );
        assert_eq!(
            node2_guard_response,
            vec![Effect::Send {
                to: NodeId(3),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: false,
                }),
            }]
        );

        assert_at_most_one_leader(&cluster);
    }

    #[test]
    fn full_quorum_overlap_sweep_never_shows_two_leaders_in_the_same_term() {
        // The most direct proof of the roadmap gate's literal wording: a
        // longer seeded schedule interleaving several add/remove
        // config-change operations amid ordinary ticks/message delivery,
        // still holding leaders().len() <= 1 and single-leader-per-term
        // throughout.
        let cluster = Cluster::new(vec![
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2), NodeId(3)], 5, 1).expect("valid node"),
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2), NodeId(3)], 5, 1).expect("valid node"),
            Node::new(NodeId(3), vec![NodeId(1), NodeId(2), NodeId(3)], 5, 1).expect("valid node"),
            // Node::new_learner doesn't exist yet -- these two stand in
            // for real joiners with the eventual full voter set
            // pre-seeded and a deliberately huge election timeout, so
            // they can never win an election on their own before being
            // genuinely admitted by the leader's replication.
            Node::new(
                NodeId(4),
                vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
                1_000_000,
                1,
            )
            .expect("valid node"),
            Node::new(
                NodeId(5),
                vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4), NodeId(5)],
                1_000_000,
                1,
            )
            .expect("valid node"),
        ])
        .expect("valid cluster");

        let mut simulation = Simulation::new(cluster, 20260820, 3, 6, 1, 2);

        let mut leader_by_term: BTreeMap<u64, NodeId> = BTreeMap::new();
        let mut added_4 = false;
        let mut added_5 = false;
        let mut removed_2 = false;

        for tick in 0..1_000u64 {
            simulation.step();

            let leaders = simulation.leaders();
            assert!(
                leaders.len() <= 1,
                "no more than one leader may exist at once, saw {leaders:?} at tick {tick}"
            );
            for &(id, term) in &leaders {
                match leader_by_term.get(&term) {
                    Some(&existing) if existing != id => {
                        panic!(
                            "two different leaders observed in term {term}: {existing:?} \
                             and {id:?} at tick {tick}"
                        );
                    }
                    _ => {
                        leader_by_term.insert(term, id);
                    }
                }
            }

            let Some(&(leader_id, _)) = leaders.first() else {
                continue;
            };
            let voters = simulation
                .node(leader_id)
                .expect("leader is a known node")
                .voters()
                .to_vec();

            if !added_4 && tick > 50 && !voters.contains(&NodeId(4)) {
                simulation.add_learner(leader_id, NodeId(4));
                let mut new_voters = voters.clone();
                new_voters.push(NodeId(4));
                new_voters.sort();
                // Gated on the call actually succeeding (Some), not merely
                // being made -- a leader that lost the role between
                // reading `leaders()` above and this call would silently
                // reject it (propose_config_change returns None for a
                // non-leader), and a flag set regardless would make the
                // final assertion below pass without the sweep having
                // done anything.
                if simulation
                    .propose_config_change(leader_id, new_voters)
                    .is_some()
                {
                    added_4 = true;
                }
            } else if added_4 && !added_5 && tick > 150 && !voters.contains(&NodeId(5)) {
                simulation.add_learner(leader_id, NodeId(5));
                let mut new_voters = voters.clone();
                new_voters.push(NodeId(5));
                new_voters.sort();
                if simulation
                    .propose_config_change(leader_id, new_voters)
                    .is_some()
                {
                    added_5 = true;
                }
            } else if added_5 && !removed_2 && tick > 250 && voters.contains(&NodeId(2)) {
                let new_voters: Vec<NodeId> =
                    voters.into_iter().filter(|&id| id != NodeId(2)).collect();
                if simulation
                    .propose_config_change(leader_id, new_voters)
                    .is_some()
                {
                    removed_2 = true;
                }
            }
        }

        assert!(
            added_4 && added_5 && removed_2,
            "the sweep must actually exercise all three config-change operations, not just \
             ticks"
        );

        // Proving the three calls fired proves nothing about whether any
        // of them actually committed -- assert real end-state convergence
        // too: node 1 was never removed, so its own current_config is the
        // final word on whether the whole sequence (add 4, add 5, remove
        // 2) genuinely replicated and committed, not just got proposed and
        // then lost to a subsequent election or dropped message.
        let final_config = simulation
            .node(NodeId(1))
            .expect("node 1 is a known node")
            .current_config()
            .clone();
        assert_eq!(
            final_config.voters,
            vec![NodeId(1), NodeId(3), NodeId(4), NodeId(5)],
            "the sweep must actually converge to the final voter set"
        );
        assert!(
            final_config.old_voters.is_none(),
            "the final config change (removing node 2) must have fully committed -- past its \
             joint phase, not just proposed"
        );
    }
}
