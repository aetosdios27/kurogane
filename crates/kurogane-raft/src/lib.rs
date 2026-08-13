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

/// A message understood by the transport-free Raft core.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Message {
    RequestVote(RequestVote),
    RequestVoteResponse(RequestVoteResponse),
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
}

impl Node {
    /// Constructs a follower in term zero from a fixed, canonical membership.
    pub fn new(id: NodeId, peers: Vec<NodeId>, election_timeout: u64) -> Result<Self, ConfigError> {
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

        Ok(Self {
            id,
            peers,
            role: Role::Follower,
            current_term: 0,
            voted_for: None,
            election_elapsed: 0,
            election_timeout,
            votes_granted: BTreeSet::new(),
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
}

#[cfg(test)]
mod tests {
    use super::{ConfigError, Node, NodeId, Role};

    #[test]
    fn constructs_initial_follower_state() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let node = Node::new(NodeId(2), peers.clone(), 11).expect("valid configuration");

        assert_eq!(node.id(), NodeId(2));
        assert_eq!(node.peers(), peers);
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.voted_for(), None);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(node.election_timeout(), 11);
        assert!(node.votes_granted().is_empty());
    }

    #[test]
    fn rejects_invalid_configurations() {
        let cases = [
            (
                Node::new(NodeId(1), vec![], 1),
                ConfigError::EmptyConfiguration,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1), NodeId(1)], 1),
                ConfigError::MembersNotStrictlyOrdered,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(2), NodeId(1)], 1),
                ConfigError::MembersNotStrictlyOrdered,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(2)], 1),
                ConfigError::LocalNodeMissing,
            ),
            (
                Node::new(NodeId(1), vec![NodeId(1)], 0),
                ConfigError::ZeroElectionTimeout,
            ),
        ];

        for (result, expected) in cases {
            assert_eq!(result.expect_err("configuration must fail"), expected);
        }
    }
}
