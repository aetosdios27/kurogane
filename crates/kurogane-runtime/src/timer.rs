//! Drives `Event::Tick` on a real wall-clock interval — the only place
//! actual time enters the system. `kurogane-raft` never reads a clock
//! itself; the same design as `kurogane-sim`'s seeded `Simulation`, just
//! backed by real time and real randomness instead of deterministic ones.

use std::time::Duration;

use crate::actor::ActorHandle;

/// Ticks `handle` every `tick_interval`, drawing a fresh randomized
/// `next_timeout` from `[min_timeout, max_timeout]` on every tick — used
/// only if the node decides to start a new election on that tick, same as
/// `Simulation::step` already does deterministically in tests.
pub async fn run(handle: ActorHandle, tick_interval: Duration, min_timeout: u64, max_timeout: u64) {
    let mut interval = tokio::time::interval(tick_interval);
    loop {
        interval.tick().await;
        let next_timeout = rand::random_range(min_timeout..=max_timeout);
        handle.tick(next_timeout);
    }
}

#[cfg(test)]
mod tests {
    use kurogane_kv::{Command, Replica};
    use kurogane_raft::{Message, Node, NodeId};
    use tempfile::tempdir;

    use super::*;
    use crate::actor::{self, Actor, PeerTransport, ProposeOutcome};
    use crate::storage::Storage;

    struct NoopTransport;

    impl PeerTransport for NoopTransport {
        fn send(&mut self, _to: NodeId, _message: Message) {}
    }

    #[tokio::test(start_paused = true)]
    async fn drives_a_single_node_cluster_to_leadership_via_real_ticks() {
        let node = Node::new(NodeId(1), vec![NodeId(1)], 3, 1).expect("valid node");
        let dir = tempdir().expect("temp dir");
        let storage = Storage::open(dir.path().join("state")).expect("open storage");
        let actor = Actor::new(Replica::new(node), storage, NoopTransport, u64::MAX);
        let (handle, receiver) = actor::channel(8);
        tokio::spawn(actor::run(actor, receiver));
        tokio::spawn(run(handle.clone(), Duration::from_millis(10), 5, 5));

        // Enough virtual ticks for the node's election_timeout (3) to
        // elapse; paused time auto-advances through the idle interval waits.
        tokio::time::sleep(Duration::from_millis(40)).await;

        let outcome = handle
            .propose(Command::Get { key: vec![1] })
            .await
            .expect("actor task is alive");
        assert!(
            matches!(outcome, ProposeOutcome::Accepted(_)),
            "a single-node cluster should have elected itself leader by now, got {outcome:?}"
        );
    }
}
