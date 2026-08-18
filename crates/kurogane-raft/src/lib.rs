//! Transport-free types and state ownership for Kurogane's Raft core.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Stable identity of one member in a Raft configuration.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NodeId(pub u64);

/// A node's current Raft role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Role {
    Follower,
    Candidate,
    Leader,
}

/// A vote request delivered by another configured node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestVote {
    pub term: u64,
    pub candidate_id: NodeId,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

/// The response to a vote request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestVoteResponse {
    pub term: u64,
    pub granted: bool,
}

/// A leader heartbeat. Milestone two carries no entries; replication is
/// milestone three's `prev_log_index`/`prev_log_term`/`entries` addition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendEntries {
    pub term: u64,
    pub leader_id: NodeId,
}

/// The response to a heartbeat. `success` has no meaning yet: there are no
/// entries to conflict against until milestone three.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendEntriesResponse {
    pub term: u64,
}

/// A message understood by the transport-free Raft core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Message {
    RequestVote(RequestVote),
    RequestVoteResponse(RequestVoteResponse),
    AppendEntries(AppendEntries),
    AppendEntriesResponse(AppendEntriesResponse),
}

/// One explicit input to a node transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Event {
    Tick { next_timeout: u64 },
    Step { from: NodeId, message: Message },
}

/// One side effect emitted by a node transition for its owner to interpret.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Effect {
    Send { to: NodeId, message: Message },
}

/// Invalid construction of a Raft node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConfigError {
    EmptyConfiguration,
    MembersNotStrictlyOrdered,
    LocalNodeMissing,
    ZeroElectionTimeout,
    ZeroHeartbeatInterval,
    HeartbeatIntervalExceedsElectionTimeout,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyConfiguration => "Raft configuration must contain at least one node",
            Self::MembersNotStrictlyOrdered => {
                "Raft configuration must be sorted and contain no duplicate node IDs"
            }
            Self::LocalNodeMissing => "Raft configuration must contain the local node ID",
            Self::ZeroElectionTimeout => "election timeout must be greater than zero",
            Self::ZeroHeartbeatInterval => "heartbeat interval must be greater than zero",
            Self::HeartbeatIntervalExceedsElectionTimeout => {
                "heartbeat interval must not exceed the election timeout"
            }
        };

        formatter.write_str(message)
    }
}

impl Error for ConfigError {}

/// All mutable protocol state owned by one Raft node.
#[derive(Debug)]
pub struct Node {
    id: NodeId,
    peers: Vec<NodeId>,
    role: Role,
    current_term: u64,
    voted_for: Option<NodeId>,
    election_elapsed: u64,
    election_timeout: u64,
    votes_granted: BTreeSet<NodeId>,
    heartbeat_elapsed: u64,
    heartbeat_interval: u64,
    last_log_index: u64,
    last_log_term: u64,
}

impl Node {
    /// Constructs a follower in term zero from a fixed, canonical membership.
    pub fn new(
        id: NodeId,
        peers: Vec<NodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
    ) -> Result<Self, ConfigError> {
        if peers.is_empty() {
            return Err(ConfigError::EmptyConfiguration);
        }
        if peers.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ConfigError::MembersNotStrictlyOrdered);
        }
        if peers.binary_search(&id).is_err() {
            return Err(ConfigError::LocalNodeMissing);
        }
        if election_timeout == 0 {
            return Err(ConfigError::ZeroElectionTimeout);
        }
        if heartbeat_interval == 0 {
            return Err(ConfigError::ZeroHeartbeatInterval);
        }
        if heartbeat_interval > election_timeout {
            return Err(ConfigError::HeartbeatIntervalExceedsElectionTimeout);
        }

        Ok(Self {
            id,
            peers,
            role: Role::Follower,
            current_term: 0,
            voted_for: None,
            election_elapsed: 0,
            election_timeout,
            votes_granted: BTreeSet::new(),
            heartbeat_elapsed: 0,
            heartbeat_interval,
            last_log_index: 0,
            last_log_term: 0,
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn peers(&self) -> &[NodeId] {
        &self.peers
    }

    pub fn role(&self) -> Role {
        self.role
    }

    pub fn current_term(&self) -> u64 {
        self.current_term
    }

    pub fn voted_for(&self) -> Option<NodeId> {
        self.voted_for
    }

    pub fn election_elapsed(&self) -> u64 {
        self.election_elapsed
    }

    pub fn election_timeout(&self) -> u64 {
        self.election_timeout
    }

    pub fn votes_granted(&self) -> &BTreeSet<NodeId> {
        &self.votes_granted
    }

    pub fn heartbeat_elapsed(&self) -> u64 {
        self.heartbeat_elapsed
    }

    pub fn heartbeat_interval(&self) -> u64 {
        self.heartbeat_interval
    }

    pub fn last_log_index(&self) -> u64 {
        self.last_log_index
    }

    pub fn last_log_term(&self) -> u64 {
        self.last_log_term
    }

    /// Applies one explicit input to this node's protocol state, returning the
    /// effects its owner must interpret (e.g. sending a message to a peer).
    pub fn step(&mut self, event: Event) -> Vec<Effect> {
        match event {
            Event::Tick { next_timeout } => self.on_tick(next_timeout),
            Event::Step { from, message } => {
                if !self.is_member(from) {
                    return Vec::new();
                }

                match message {
                    Message::RequestVote(request) => {
                        if request.candidate_id != from {
                            return Vec::new();
                        }
                        self.on_request_vote(from, request)
                    }
                    Message::RequestVoteResponse(response) => {
                        self.on_request_vote_response(from, response)
                    }
                    Message::AppendEntries(request) => {
                        if request.leader_id != from {
                            return Vec::new();
                        }
                        self.on_append_entries(from, request)
                    }
                    Message::AppendEntriesResponse(response) => {
                        self.on_append_entries_response(response);
                        Vec::new()
                    }
                }
            }
        }
    }

    fn is_member(&self, id: NodeId) -> bool {
        self.peers.binary_search(&id).is_ok()
    }

    fn on_tick(&mut self, next_timeout: u64) -> Vec<Effect> {
        if self.role == Role::Leader {
            return self.on_leader_tick();
        }

        self.election_elapsed += 1;
        if self.election_elapsed < self.election_timeout {
            return Vec::new();
        }

        self.start_election(next_timeout)
    }

    fn on_leader_tick(&mut self) -> Vec<Effect> {
        self.heartbeat_elapsed += 1;
        if self.heartbeat_elapsed < self.heartbeat_interval {
            return Vec::new();
        }

        self.heartbeat_elapsed = 0;
        self.send_heartbeats()
    }

    fn send_heartbeats(&self) -> Vec<Effect> {
        self.peers
            .iter()
            .copied()
            .filter(|&peer| peer != self.id)
            .map(|peer| Effect::Send {
                to: peer,
                message: Message::AppendEntries(AppendEntries {
                    term: self.current_term,
                    leader_id: self.id,
                }),
            })
            .collect()
    }

    fn start_election(&mut self, next_timeout: u64) -> Vec<Effect> {
        self.role = Role::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.votes_granted.clear();
        self.votes_granted.insert(self.id);
        self.election_elapsed = 0;
        self.election_timeout = next_timeout;

        if self.has_quorum() {
            self.role = Role::Leader;
            self.heartbeat_elapsed = 0;
            return self.send_heartbeats();
        }

        self.peers
            .iter()
            .copied()
            .filter(|&peer| peer != self.id)
            .map(|peer| Effect::Send {
                to: peer,
                message: Message::RequestVote(RequestVote {
                    term: self.current_term,
                    candidate_id: self.id,
                    last_log_index: self.last_log_index,
                    last_log_term: self.last_log_term,
                }),
            })
            .collect()
    }

    fn on_request_vote(&mut self, from: NodeId, request: RequestVote) -> Vec<Effect> {
        if request.term < self.current_term {
            return vec![Effect::Send {
                to: from,
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: self.current_term,
                    granted: false,
                }),
            }];
        }

        if request.term > self.current_term {
            self.step_down(request.term);
        }

        let log_is_up_to_date = (request.last_log_term, request.last_log_index)
            >= (self.last_log_term, self.last_log_index);
        let already_voted_for_candidate = self.voted_for == Some(request.candidate_id);
        let can_grant =
            log_is_up_to_date && (self.voted_for.is_none() || already_voted_for_candidate);

        if can_grant {
            self.voted_for = Some(request.candidate_id);
            self.election_elapsed = 0;
        }

        vec![Effect::Send {
            to: from,
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: self.current_term,
                granted: can_grant,
            }),
        }]
    }

    fn on_request_vote_response(
        &mut self,
        from: NodeId,
        response: RequestVoteResponse,
    ) -> Vec<Effect> {
        if response.term > self.current_term {
            self.step_down(response.term);
            return Vec::new();
        }

        if response.term < self.current_term || self.role != Role::Candidate {
            return Vec::new();
        }

        if response.granted {
            self.votes_granted.insert(from);
            if self.has_quorum() {
                self.role = Role::Leader;
                self.heartbeat_elapsed = 0;
                return self.send_heartbeats();
            }
        }

        Vec::new()
    }

    fn on_append_entries(&mut self, from: NodeId, request: AppendEntries) -> Vec<Effect> {
        if request.term < self.current_term {
            return vec![Effect::Send {
                to: from,
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: self.current_term,
                }),
            }];
        }

        if request.term > self.current_term {
            self.step_down(request.term);
        } else if self.role == Role::Candidate {
            self.role = Role::Follower;
        }

        self.election_elapsed = 0;

        vec![Effect::Send {
            to: from,
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: self.current_term,
            }),
        }]
    }

    fn on_append_entries_response(&mut self, response: AppendEntriesResponse) {
        if response.term > self.current_term {
            self.step_down(response.term);
        }
    }

    fn step_down(&mut self, term: u64) {
        self.role = Role::Follower;
        self.current_term = term;
        self.voted_for = None;
        self.votes_granted.clear();
    }

    fn quorum_size(&self) -> usize {
        self.peers.len() / 2 + 1
    }

    fn has_quorum(&self) -> bool {
        self.votes_granted.len() >= self.quorum_size()
    }

    /// Test-only hook for exercising log-freshness scenarios before milestone
    /// three gives nodes a real log. Not part of the public API.
    #[cfg(test)]
    fn set_log_tip(&mut self, last_log_index: u64, last_log_term: u64) {
        self.last_log_index = last_log_index;
        self.last_log_term = last_log_term;
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        AppendEntries, AppendEntriesResponse, ConfigError, Effect, Event, Message, Node, NodeId,
        RequestVote, RequestVoteResponse, Role,
    };

    #[test]
    fn constructs_initial_follower_state() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let node = Node::new(NodeId(2), peers.clone(), 11, 4).expect("valid configuration");

        assert_eq!(node.id(), NodeId(2));
        assert_eq!(node.peers(), peers);
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.voted_for(), None);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(node.election_timeout(), 11);
        assert_eq!(node.heartbeat_elapsed(), 0);
        assert_eq!(node.heartbeat_interval(), 4);
        assert_eq!(node.last_log_index(), 0);
        assert_eq!(node.last_log_term(), 0);
        assert!(node.votes_granted().is_empty());
    }

    #[test]
    fn rejects_invalid_configurations() {
        let cases = [
            (
                Node::new(NodeId(1), vec![], 1, 1),
                ConfigError::EmptyConfiguration,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1), NodeId(1)], 1, 1),
                ConfigError::MembersNotStrictlyOrdered,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(2), NodeId(1)], 1, 1),
                ConfigError::MembersNotStrictlyOrdered,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(2)], 1, 1),
                ConfigError::LocalNodeMissing,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1)], 0, 1),
                ConfigError::ZeroElectionTimeout,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1)], 1, 0),
                ConfigError::ZeroHeartbeatInterval,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1)], 1, 2),
                ConfigError::HeartbeatIntervalExceedsElectionTimeout,
            ),
        ];

        for (result, expected) in cases {
            assert_eq!(result.expect_err("configuration must fail"), expected);
        }
    }

    #[test]
    fn tick_below_timeout_produces_no_effects() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 3, 1).expect("valid node");

        let effects = node.step(Event::Tick { next_timeout: 5 });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.election_elapsed(), 1);
    }

    #[test]
    fn tick_at_timeout_starts_election_and_requests_votes_from_peers() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");

        let effects = node.step(Event::Tick { next_timeout: 7 });

        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.voted_for(), Some(NodeId(1)));
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(node.election_timeout(), 7);
        assert_eq!(node.votes_granted(), &BTreeSet::from([NodeId(1)]));

        let expected_request = RequestVote {
            term: 1,
            candidate_id: NodeId(1),
            last_log_index: 0,
            last_log_term: 0,
        };
        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::RequestVote(expected_request),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::RequestVote(expected_request),
                },
            ]
        );
    }

    #[test]
    fn single_node_cluster_wins_election_immediately() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");

        let effects = node.step(Event::Tick { next_timeout: 5 });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Leader);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.heartbeat_elapsed(), 0);
    }

    #[test]
    fn grants_vote_and_resets_election_elapsed() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(2),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(node.voted_for(), Some(NodeId(2)));
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(node.current_term(), 1);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: true,
                }),
            }]
        );
    }

    #[test]
    fn rejects_vote_when_already_voted_for_different_candidate() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(2),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(node.voted_for(), Some(NodeId(2)));
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
    fn rejects_request_vote_with_stale_term() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 0,
                candidate_id: NodeId(2),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.voted_for(), Some(NodeId(1)));
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
    fn steps_down_and_grants_vote_on_higher_term_request_vote() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 2,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 2);
        assert_eq!(node.voted_for(), Some(NodeId(3)));
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(3),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 2,
                    granted: true,
                }),
            }]
        );
    }

    #[test]
    fn becomes_leader_and_sends_an_immediate_heartbeat_once_quorum_is_granted() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 3, 3).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Candidate);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });

        assert_eq!(node.role(), Role::Leader);
        assert_eq!(node.heartbeat_elapsed(), 0);
        assert_eq!(
            node.votes_granted(),
            &BTreeSet::from([NodeId(1), NodeId(2)])
        );
        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntries(AppendEntries {
                        term: 1,
                        leader_id: NodeId(1),
                    }),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::AppendEntries(AppendEntries {
                        term: 1,
                        leader_id: NodeId(1),
                    }),
                },
            ]
        );
    }

    #[test]
    fn leader_sends_heartbeats_only_once_the_interval_elapses() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 3, 3).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        assert!(node.step(Event::Tick { next_timeout: 5 }).is_empty());
        assert!(node.step(Event::Tick { next_timeout: 5 }).is_empty());

        let effects = node.step(Event::Tick { next_timeout: 5 });

        assert_eq!(node.heartbeat_elapsed(), 0);
        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntries(AppendEntries {
                        term: 1,
                        leader_id: NodeId(1),
                    }),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::AppendEntries(AppendEntries {
                        term: 1,
                        leader_id: NodeId(1),
                    }),
                },
            ]
        );
    }

    #[test]
    fn follower_resets_election_elapsed_on_valid_append_entries() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.election_elapsed(), 1);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse { term: 1 }),
            }]
        );
    }

    #[test]
    fn candidate_steps_down_on_same_term_append_entries() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Candidate);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse { term: 1 }),
            }]
        );
    }

    #[test]
    fn rejects_stale_term_append_entries() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.current_term(), 1);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 0,
                leader_id: NodeId(2),
            }),
        });

        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 1);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse { term: 1 }),
            }]
        );
    }

    #[test]
    fn leader_steps_down_on_higher_term_append_entries() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::AppendEntries(AppendEntries {
                term: 2,
                leader_id: NodeId(3),
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 2);
        assert_eq!(node.voted_for(), None);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(3),
                message: Message::AppendEntriesResponse(AppendEntriesResponse { term: 2 }),
            }]
        );
    }

    #[test]
    fn leader_steps_down_on_higher_term_append_entries_response() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::AppendEntriesResponse(AppendEntriesResponse { term: 4 }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 4);
        assert_eq!(node.voted_for(), None);
    }

    #[test]
    fn ignores_append_entries_whose_leader_id_does_not_match_the_sender() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(3),
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.role(), Role::Follower);
    }

    #[test]
    fn grants_vote_when_candidate_log_is_at_least_as_up_to_date() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.set_log_tip(4, 2);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(2),
                last_log_index: 4,
                last_log_term: 2,
            }),
        });

        assert_eq!(node.voted_for(), Some(NodeId(2)));
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: true,
                }),
            }]
        );
    }

    #[test]
    fn rejects_vote_when_candidate_log_is_behind_even_after_stepping_down_on_a_higher_term() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.set_log_tip(4, 3);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 9,
                candidate_id: NodeId(2),
                last_log_index: 1,
                last_log_term: 2,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 9);
        assert_eq!(node.voted_for(), None);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 9,
                    granted: false,
                }),
            }]
        );
    }

    #[test]
    fn steps_down_on_higher_term_vote_response() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 5,
                granted: false,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 5);
        assert_eq!(node.voted_for(), None);
        assert!(node.votes_granted().is_empty());
    }

    #[test]
    fn ignores_stale_term_vote_response() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 1 });
        node.step(Event::Tick { next_timeout: 1 });
        assert_eq!(node.current_term(), 2);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 2);
        assert_eq!(node.votes_granted(), &BTreeSet::from([NodeId(1)]));
    }

    #[test]
    fn ignores_request_vote_from_a_nonmember_node_id() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.step(Event::Step {
            from: NodeId(9),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(9),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.voted_for(), None);
    }

    #[test]
    fn ignores_request_vote_whose_candidate_id_does_not_match_the_sender() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVote(RequestVote {
                term: 1,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.voted_for(), None);
    }

    #[test]
    fn a_nonmember_vote_response_never_counts_toward_quorum() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Candidate);

        let effects = node.step(Event::Step {
            from: NodeId(9),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.votes_granted(), &BTreeSet::from([NodeId(1)]));
    }

    #[test]
    fn leader_ignores_tick_events() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Leader);

        let effects = node.step(Event::Tick { next_timeout: 9 });

        assert!(effects.is_empty());
        assert_eq!(node.role(), Role::Leader);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(node.election_timeout(), 5);
    }
}
