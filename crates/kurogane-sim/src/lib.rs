//! Deterministic ownership of Raft nodes for controlled simulations.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use kurogane_raft::{Node, NodeId};

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

#[cfg(test)]
mod tests {
    use kurogane_raft::{Node, NodeId};

    use super::{Cluster, ClusterError};

    fn node(id: u64, peers: &[NodeId]) -> Node {
        Node::new(NodeId(id), peers.to_vec(), 10 + id).expect("valid node")
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
