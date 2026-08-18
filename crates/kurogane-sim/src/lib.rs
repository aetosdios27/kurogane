//! Deterministic ownership of Raft nodes for controlled simulations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use kurogane_raft::{Effect, Event, Message, Node, NodeId, Role};

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
            .peers()
            .to_vec();

        for node in by_id.values() {
            if node.peers() != expected_members {
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
        for Effect::Send { to, message } in effects {
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
    use kurogane_raft::{Node, NodeId};

    use super::{Cluster, ClusterError};

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
}

#[cfg(test)]
mod simulation_tests {
    use std::collections::BTreeMap;

    use kurogane_raft::{
        AppendEntries, AppendEntriesResponse, Effect, Event, Message, Node, NodeId, RequestVote,
        RequestVoteResponse, Role,
    };

    use super::{Cluster, Simulation};

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
        for Effect::Send { to, message } in requests {
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for Effect::Send {
                to: respond_to,
                message,
            } in responses
            {
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
        for Effect::Send { to, message } in node2_requests {
            if to != NodeId(3) {
                continue;
            }
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(2),
                message,
            });
            for Effect::Send {
                to: respond_to,
                message,
            } in responses
            {
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
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: 2,
                    success: true,
                    match_index: 0,
                }),
            }]
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
            requests.extend(
                effects
                    .into_iter()
                    .map(|Effect::Send { to, message }| (id, to, message)),
            );
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
            for Effect::Send {
                to: respond_to,
                message,
            } in responses
            {
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

        for Effect::Send { to, message } in retry_effects {
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for Effect::Send {
                to: respond_to,
                message,
            } in responses
            {
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
        for Effect::Send { to, message } in requests {
            let responses = cluster.node_mut(to).expect("known node").step(Event::Step {
                from: NodeId(1),
                message,
            });
            for Effect::Send {
                to: respond_to,
                message,
            } in responses
            {
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
}
