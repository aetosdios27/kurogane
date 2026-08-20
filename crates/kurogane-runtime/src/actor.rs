//! The core Tokio task: owns one `Replica` exclusively (mirrors
//! `decisions.md`'s "one authoritative node state owner" — concurrency is
//! serialized here, never inside `kurogane-raft`/`kurogane-kv`), dispatches
//! its effects, and tracks a best-effort leader hint for client redirection.

use std::io;

use kurogane_kv::{ApplyResult, Command, Replica, StateMachine};
use kurogane_raft::{Effect, Event, Message, Node, NodeId, Role};

use crate::storage::Storage;

/// How the actor gets a `Send` effect's message to a peer. Implementations
/// must not block the caller on the network — queue it and return; a peer
/// that's slow or unreachable must never stall the core loop. Dropping a
/// message here is safe: Raft's own retry logic (the periodic heartbeat
/// cycle) is what actually guarantees eventual delivery, not this trait.
pub trait PeerTransport {
    fn send(&mut self, to: NodeId, message: Message);

    /// Registers (or replaces) one peer's outbound address, e.g. once a
    /// real `AddLearner` RPC introduces a new server this node didn't know
    /// about at startup. Defaults to a no-op that reports success: a
    /// transport that routes by some other means (or a test double that
    /// never needs a real connection) shouldn't have to implement this just
    /// to compile. Returns whether `address` was accepted as reachable.
    fn add_peer(&mut self, _id: NodeId, _address: String) -> bool {
        true
    }
}

/// The result of a `propose` or `propose_config_change` call: accepted at
/// this log index, or a redirect hint (if one is known) when this node
/// isn't the leader. Carrying the hint here means a rejected client doesn't
/// need a second round trip just to ask who the leader is. Shared between
/// the two calls since both mirror the exact same "index or redirect"
/// shape at the `Node` layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProposeOutcome {
    Accepted(u64),
    NotLeader(Option<NodeId>),
}

/// The result of an `add_learner` call. Unlike `ProposeOutcome`, there's no
/// log index to report -- `Node::add_learner` doesn't append anything --
/// so `Accepted` instead carries whether the transport accepted `address`
/// as a valid endpoint. That's a distinct concern from whether the learner
/// has actually caught up yet, which the caller must poll for separately
/// (e.g. via its own storage file, the way this project's tests already do
/// for ordinary replication catch-up).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AddLearnerOutcome {
    Accepted { connected: bool },
    NotLeader(Option<NodeId>),
}

/// Drives one `Replica`, persisting and dispatching the effects it returns
/// in order — a `Persist*` effect is always applied before any `Send` after
/// it in the same batch, matching `kurogane-raft`'s write-before-send
/// contract.
pub struct Actor<T: PeerTransport> {
    replica: Replica,
    storage: Storage,
    transport: T,
    leader_hint: Option<NodeId>,
    /// Triggers compaction once this many entries have been applied since
    /// the last one. Kept deliberately small in tests to force real
    /// compaction without needing hundreds of writes.
    compact_threshold: u64,
}

impl<T: PeerTransport> Actor<T> {
    pub fn new(replica: Replica, storage: Storage, transport: T, compact_threshold: u64) -> Self {
        Self {
            replica,
            storage,
            transport,
            leader_hint: None,
            compact_threshold,
        }
    }

    pub fn node(&self) -> &Node {
        self.replica.node()
    }

    pub fn state_machine(&self) -> &StateMachine {
        self.replica.state_machine()
    }

    pub fn applied_result(&self, index: u64) -> Option<&ApplyResult> {
        self.replica.applied_result(index)
    }

    /// The last leader this node has seen at a current-or-newer term, for
    /// redirecting a client `Propose` sent to the wrong node. Best-effort:
    /// it can go stale the moment the real leader changes, same as any
    /// cached hint in a distributed system — a client that acts on a stale
    /// hint just gets another (fresher) one back.
    pub fn leader_hint(&self) -> Option<NodeId> {
        self.leader_hint
    }

    /// For a `Tick`, or an incoming peer *response*
    /// (`RequestVoteResponse`/`AppendEntriesResponse`) — there's no specific
    /// reply target for either, so every resulting effect (including a
    /// retry `Send`) is dispatched normally.
    pub fn handle_event(&mut self, event: Event) -> io::Result<()> {
        if let Event::Step { from, message } = &event {
            self.update_leader_hint(*from, message);
        }
        let effects = self.replica.step(event);
        self.dispatch(effects)?;
        self.maybe_compact()
    }

    /// For an incoming peer *request* (`RequestVote`/`AppendEntries`)
    /// arriving over one specific RPC call: persists everything as usual,
    /// but returns the response message for the caller to reply with
    /// directly, rather than dispatching it through `PeerTransport` — it's
    /// a reply to this exact call, not a fresh outbound send.
    ///
    /// `Ok(None)` means `kurogane-raft` silently dropped the request (an
    /// unrecognized `from`, or a `candidate_id`/`leader_id` that doesn't
    /// match it) — that's untrusted network input, not a bug, so the caller
    /// gets a clean "no reply" instead of this panicking. A message that
    /// *did* pass those checks always ends in exactly one `Send`; only that
    /// case would be an internal invariant violation worth panicking over.
    pub fn handle_peer_request(
        &mut self,
        from: NodeId,
        message: Message,
    ) -> io::Result<Option<Message>> {
        self.update_leader_hint(from, &message);

        let mut effects = self.replica.step(Event::Step { from, message });
        let Some(last) = effects.pop() else {
            return Ok(None);
        };
        let reply = match last {
            Effect::Send { message, .. } => message,
            _ => panic!(
                "a RequestVote/AppendEntries request that produced any effects must end with exactly one Send"
            ),
        };
        for effect in &effects {
            if let Effect::PersistHardState { .. }
            | Effect::PersistLog { .. }
            | Effect::PersistSnapshot { .. } = effect
            {
                self.storage.apply(effect)?;
            }
        }
        self.maybe_compact()?;
        Ok(Some(reply))
    }

    /// Proposes `command` if this node is the leader, returning its log
    /// index. If it isn't, returns the current `leader_hint` so the caller
    /// can redirect without a second round trip.
    pub fn propose(&mut self, command: Command) -> io::Result<ProposeOutcome> {
        let Some((index, effects)) = self.replica.propose(command) else {
            return Ok(ProposeOutcome::NotLeader(self.leader_hint));
        };
        self.dispatch(effects)?;
        self.maybe_compact()?;
        Ok(ProposeOutcome::Accepted(index))
    }

    /// Begins a joint-consensus transition to `new_voters` if this node is
    /// the leader, mirroring `propose` exactly.
    pub fn propose_config_change(&mut self, new_voters: Vec<NodeId>) -> io::Result<ProposeOutcome> {
        let Some((index, effects)) = self.replica.propose_config_change(new_voters) else {
            return Ok(ProposeOutcome::NotLeader(self.leader_hint));
        };
        self.dispatch(effects)?;
        self.maybe_compact()?;
        Ok(ProposeOutcome::Accepted(index))
    }

    /// Registers `id` as a non-voting learner reachable at `address`, if
    /// this node is the leader.
    ///
    /// Order is deliberate: the raft-level registration (and its durable
    /// `PersistLearners`) happens *before* the transport is wired up. A
    /// crash between the two steps then leaves, at worst, a
    /// registered-but-unreachable learner -- harmless, since ordinary
    /// replication retries simply keep failing to reach it until an
    /// operator retries `AddLearner`. The reverse order would risk a
    /// reachable transport target that the raft layer's durable state never
    /// accounted for, which is the worse failure mode: traffic could reach
    /// a node nothing durable explains it being sent to.
    ///
    /// `Node::add_learner` returns empty effects for three different
    /// reasons (not leader, already tracked, already a voter) -- only the
    /// first is a real `NotLeader`, so role is checked explicitly rather
    /// than inferred from an empty effects list, or a legitimate idempotent
    /// retry against the real leader would misreport itself as a redirect.
    pub fn add_learner(&mut self, id: NodeId, address: String) -> io::Result<AddLearnerOutcome> {
        if self.replica.node().role() != Role::Leader {
            return Ok(AddLearnerOutcome::NotLeader(self.leader_hint));
        }

        let effects = self.replica.add_learner(id);
        self.dispatch(effects)?;
        self.maybe_compact()?;

        let connected = self.transport.add_peer(id, address);
        Ok(AddLearnerOutcome::Accepted { connected })
    }

    /// Stops tracking `id` as a learner, if this node is the leader. No
    /// gRPC surface exposes this yet (promotion via `propose_config_change`
    /// already drops a promoted learner on its own) -- kept here for parity
    /// with `add_learner`/`propose_config_change` and as the natural place
    /// for future operator tooling to hang off.
    pub fn remove_learner(&mut self, id: NodeId) -> io::Result<()> {
        let effects = self.replica.remove_learner(id);
        self.dispatch(effects)?;
        self.maybe_compact()
    }

    fn update_leader_hint(&mut self, from: NodeId, message: &Message) {
        // Both a real AppendEntries and an InstallSnapshot are equally
        // good evidence of who's leading; mirrors on_append_entries's own
        // stale-term rejection: a request that would itself be rejected
        // tells us nothing about who's actually leading right now.
        let (leader_id, term) = match message {
            Message::AppendEntries(request) => (request.leader_id, request.term),
            Message::InstallSnapshot(request) => (request.leader_id, request.term),
            _ => return,
        };
        if leader_id == from && term >= self.replica.node().current_term() {
            self.leader_hint = Some(leader_id);
        }
    }

    /// Compacts once at least `compact_threshold` entries have been
    /// applied since the last compaction, retaining half that margin so a
    /// peer only briefly behind still catches up via an ordinary
    /// `AppendEntries` instead of a full snapshot transfer. Retaining
    /// strictly less than the threshold is load-bearing: retaining at
    /// least as much as the trigger threshold would mean `up_to_index`
    /// never clears the current boundary, and compaction would silently
    /// never advance.
    fn maybe_compact(&mut self) -> io::Result<()> {
        let applied_since_boundary = self.replica.state_machine().last_applied()
            - self.replica.node().snapshot().last_included_index;
        if applied_since_boundary < self.compact_threshold {
            return Ok(());
        }
        if let Some(effects) = self.replica.compact(self.compact_threshold / 2) {
            self.dispatch(effects)?;
        }
        Ok(())
    }

    fn dispatch(&mut self, effects: Vec<Effect>) -> io::Result<()> {
        for effect in &effects {
            match effect {
                Effect::PersistHardState { .. }
                | Effect::PersistLog { .. }
                | Effect::PersistSnapshot { .. }
                | Effect::PersistLearners { .. } => {
                    self.storage.apply(effect)?;
                }
                Effect::Send { to, message } => {
                    self.transport.send(*to, message.clone());
                }
            }
        }
        Ok(())
    }
}

/// One request to the actor's task, routed through an `ActorHandle`.
enum ActorRequest {
    Tick {
        next_timeout: u64,
    },
    PeerRequest {
        from: NodeId,
        message: Message,
        reply: tokio::sync::oneshot::Sender<Option<Message>>,
    },
    PeerResponse {
        from: NodeId,
        message: Message,
    },
    Propose {
        command: Command,
        reply: tokio::sync::oneshot::Sender<ProposeOutcome>,
    },
    ProposeConfigChange {
        new_voters: Vec<NodeId>,
        reply: tokio::sync::oneshot::Sender<ProposeOutcome>,
    },
    AddLearner {
        id: NodeId,
        address: String,
        reply: tokio::sync::oneshot::Sender<AddLearnerOutcome>,
    },
    RemoveLearner {
        id: NodeId,
        reply: tokio::sync::oneshot::Sender<()>,
    },
}

/// A cheaply cloneable way to submit work to an actor running in its own
/// task. The `Actor` itself is never shared across tasks directly — this is
/// the one door in, matching `decisions.md`'s "one authoritative node state
/// owner."
#[derive(Clone)]
pub struct ActorHandle {
    sender: tokio::sync::mpsc::Sender<ActorRequest>,
}

impl ActorHandle {
    /// Submits an incoming peer request and awaits its response.
    /// Backpressure applies here (this is a real RPC that needs an answer),
    /// unlike `tick`/`peer_response`, which are fire-and-forget. Outer
    /// `None` means the actor task is gone; inner `None` means
    /// `kurogane-raft` dropped the request (see `Actor::handle_peer_request`).
    pub async fn peer_request(&self, from: NodeId, message: Message) -> Option<Option<Message>> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ActorRequest::PeerRequest {
                from,
                message,
                reply,
            })
            .await
            .ok()?;
        receiver.await.ok()
    }

    /// Submits an incoming peer response. Fire-and-forget: dropped under
    /// backpressure, same as any other `Send` effect — Raft's own retry
    /// logic covers it.
    pub fn peer_response(&self, from: NodeId, message: Message) {
        let _ = self
            .sender
            .try_send(ActorRequest::PeerResponse { from, message });
    }

    /// Submits a client propose request and awaits its result. `None` means
    /// the actor task is gone; otherwise the outcome carries either the
    /// accepted index or a leader hint for redirection.
    pub async fn propose(&self, command: Command) -> Option<ProposeOutcome> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ActorRequest::Propose { command, reply })
            .await
            .ok()?;
        receiver.await.ok()
    }

    /// Submits a logical tick. Fire-and-forget, same reasoning as
    /// `peer_response`.
    pub fn tick(&self, next_timeout: u64) {
        let _ = self.sender.try_send(ActorRequest::Tick { next_timeout });
    }

    /// Submits a client config-change request and awaits its result. `None`
    /// means the actor task is gone; otherwise the outcome carries either
    /// the accepted index or a leader hint for redirection, same shape as
    /// `propose`.
    pub async fn propose_config_change(&self, new_voters: Vec<NodeId>) -> Option<ProposeOutcome> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ActorRequest::ProposeConfigChange { new_voters, reply })
            .await
            .ok()?;
        receiver.await.ok()
    }

    /// Submits a client `AddLearner` request and awaits its result. `None`
    /// means the actor task is gone.
    pub async fn add_learner(&self, id: NodeId, address: String) -> Option<AddLearnerOutcome> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ActorRequest::AddLearner { id, address, reply })
            .await
            .ok()?;
        receiver.await.ok()
    }

    /// Submits a `RemoveLearner` request and awaits its completion. `None`
    /// means the actor task is gone. No gRPC surface calls this yet -- see
    /// `Actor::remove_learner`'s own doc comment.
    pub async fn remove_learner(&self, id: NodeId) -> Option<()> {
        let (reply, receiver) = tokio::sync::oneshot::channel();
        self.sender
            .send(ActorRequest::RemoveLearner { id, reply })
            .await
            .ok()?;
        receiver.await.ok()
    }
}

/// The receiving half of an actor's channel. Opaque on purpose — only
/// `run` drains it, so `ActorRequest` itself never needs to be public.
pub struct ActorReceiver {
    receiver: tokio::sync::mpsc::Receiver<ActorRequest>,
}

/// Creates an actor task's channel pair: the `ActorHandle` other tasks use
/// to submit work, and the receiver `run` drains.
pub fn channel(capacity: usize) -> (ActorHandle, ActorReceiver) {
    let (sender, receiver) = tokio::sync::mpsc::channel(capacity);
    (ActorHandle { sender }, ActorReceiver { receiver })
}

/// Runs `actor` until its channel closes, draining one request at a time —
/// this is what makes it the single authoritative owner of the underlying
/// `Replica`. Panics if a `Persist*` effect fails to write durably rather
/// than silently continuing in a state that might violate the
/// write-before-send contract; graceful disk-failure handling is out of
/// scope for this milestone.
pub async fn run<T: PeerTransport>(mut actor: Actor<T>, mut requests: ActorReceiver) {
    while let Some(request) = requests.receiver.recv().await {
        match request {
            ActorRequest::Tick { next_timeout } => {
                actor
                    .handle_event(Event::Tick { next_timeout })
                    .expect("durable storage write must succeed");
            }
            ActorRequest::PeerRequest {
                from,
                message,
                reply,
            } => {
                let response = actor
                    .handle_peer_request(from, message)
                    .expect("durable storage write must succeed");
                let _ = reply.send(response);
            }
            ActorRequest::PeerResponse { from, message } => {
                actor
                    .handle_event(Event::Step { from, message })
                    .expect("durable storage write must succeed");
            }
            ActorRequest::Propose { command, reply } => {
                let result = actor
                    .propose(command)
                    .expect("durable storage write must succeed");
                let _ = reply.send(result);
            }
            ActorRequest::ProposeConfigChange { new_voters, reply } => {
                let result = actor
                    .propose_config_change(new_voters)
                    .expect("durable storage write must succeed");
                let _ = reply.send(result);
            }
            ActorRequest::AddLearner { id, address, reply } => {
                let result = actor
                    .add_learner(id, address)
                    .expect("durable storage write must succeed");
                let _ = reply.send(result);
            }
            ActorRequest::RemoveLearner { id, reply } => {
                actor
                    .remove_learner(id)
                    .expect("durable storage write must succeed");
                let _ = reply.send(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use kurogane_raft::{NodeId as RaftNodeId, RequestVoteResponse};
    use tempfile::tempdir;

    use super::*;

    struct RecordingTransport {
        sent: Vec<(NodeId, Message)>,
    }

    impl RecordingTransport {
        fn new() -> Self {
            Self { sent: Vec::new() }
        }
    }

    impl PeerTransport for RecordingTransport {
        fn send(&mut self, to: NodeId, message: Message) {
            self.sent.push((to, message));
        }
    }

    fn actor(
        id: RaftNodeId,
        peers: Vec<RaftNodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
    ) -> Actor<RecordingTransport> {
        let node = kurogane_raft::Node::new(id, peers, election_timeout, heartbeat_interval)
            .expect("valid node");
        let dir = tempdir().expect("temp dir");
        // Leaked deliberately: the tempdir must outlive the Storage, and
        // these are short-lived unit tests, not a long-running process.
        let path = Box::leak(Box::new(dir)).path().join("state");
        let storage = Storage::open(path).expect("open storage");
        // Effectively disables compaction: nothing in these short-lived
        // unit tests applies anywhere near this many entries.
        Actor::new(
            Replica::new(node),
            storage,
            RecordingTransport::new(),
            u64::MAX,
        )
    }

    #[test]
    fn tick_that_starts_an_election_persists_and_sends_request_votes() {
        let peers = vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)];
        let mut actor = actor(RaftNodeId(1), peers, 1, 1);

        actor
            .handle_event(Event::Tick { next_timeout: 5 })
            .expect("handle event");

        assert_eq!(actor.node().role(), kurogane_raft::Role::Candidate);
        assert_eq!(actor.node().current_term(), 1);
        assert_eq!(
            actor.transport.sent,
            vec![
                (
                    RaftNodeId(2),
                    Message::RequestVote(kurogane_raft::RequestVote {
                        term: 1,
                        candidate_id: RaftNodeId(1),
                        last_log_index: 0,
                        last_log_term: 0,
                    })
                ),
                (
                    RaftNodeId(3),
                    Message::RequestVote(kurogane_raft::RequestVote {
                        term: 1,
                        candidate_id: RaftNodeId(1),
                        last_log_index: 0,
                        last_log_term: 0,
                    })
                ),
            ]
        );
    }

    #[test]
    fn propose_on_a_single_node_cluster_commits_and_applies_immediately() {
        let mut actor = actor(RaftNodeId(1), vec![RaftNodeId(1)], 1, 1);
        actor
            .handle_event(Event::Tick { next_timeout: 5 })
            .expect("handle event");
        assert_eq!(actor.node().role(), kurogane_raft::Role::Leader);

        let outcome = actor
            .propose(Command::Set {
                key: vec![1],
                value: vec![9],
            })
            .expect("propose");
        let ProposeOutcome::Accepted(index) = outcome else {
            panic!("leader must accept propose, got {outcome:?}");
        };

        assert_eq!(
            actor.applied_result(index),
            Some(&ApplyResult::Set { previous: None })
        );
        assert_eq!(actor.state_machine().get(&[1]), Some(&[9][..]));
    }

    #[test]
    fn maybe_compact_fires_once_the_threshold_is_crossed_and_persists_the_snapshot() {
        let node =
            kurogane_raft::Node::new(RaftNodeId(1), vec![RaftNodeId(1)], 1, 1).expect("valid node");
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("state");
        let storage = Storage::open(&path).expect("open storage");
        let mut actor = Actor::new(Replica::new(node), storage, RecordingTransport::new(), 2);

        actor
            .handle_event(Event::Tick { next_timeout: 5 })
            .expect("handle event");
        assert_eq!(actor.node().role(), kurogane_raft::Role::Leader);

        for key in 0..3u8 {
            actor
                .propose(Command::Set {
                    key: vec![key],
                    value: vec![key],
                })
                .expect("propose");
        }

        let boundary = actor.node().snapshot().last_included_index;
        assert!(boundary > 0, "compaction should have fired by now");

        let reopened = Storage::open(&path).expect("reopen storage");
        assert_eq!(reopened.snapshot().last_included_index, boundary);
        assert_eq!(reopened.snapshot_data(), actor.node().snapshot_data());
        // The full path (compact -> dispatch -> Storage::apply) must still
        // leave the log's absolute indexing correct above the boundary,
        // not just the snapshot fields.
        assert_eq!(reopened.log().len() as u64, 3 - boundary);
    }

    #[test]
    fn propose_on_a_follower_returns_the_leader_hint() {
        let mut actor = actor(
            RaftNodeId(1),
            vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)],
            5,
            2,
        );

        let outcome = actor
            .propose(Command::Set {
                key: vec![1],
                value: vec![2],
            })
            .expect("propose");

        assert_eq!(outcome, ProposeOutcome::NotLeader(None));
    }

    #[test]
    fn valid_append_entries_updates_the_leader_hint() {
        let mut actor = actor(
            RaftNodeId(1),
            vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)],
            5,
            2,
        );
        assert_eq!(actor.leader_hint(), None);

        actor
            .handle_event(Event::Step {
                from: RaftNodeId(2),
                message: Message::AppendEntries(kurogane_raft::AppendEntries {
                    term: 1,
                    leader_id: RaftNodeId(2),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                }),
            })
            .expect("handle event");

        assert_eq!(actor.leader_hint(), Some(RaftNodeId(2)));
    }

    #[test]
    fn a_stale_term_append_entries_does_not_update_the_leader_hint() {
        let mut actor = actor(
            RaftNodeId(1),
            vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)],
            1,
            1,
        );
        actor
            .handle_event(Event::Tick { next_timeout: 5 })
            .expect("handle event");
        assert_eq!(actor.node().current_term(), 1);

        actor
            .handle_event(Event::Step {
                from: RaftNodeId(3),
                message: Message::AppendEntries(kurogane_raft::AppendEntries {
                    term: 0,
                    leader_id: RaftNodeId(3),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                }),
            })
            .expect("handle event");

        assert_eq!(actor.leader_hint(), None);
    }

    #[test]
    fn granting_a_vote_persists_hard_state_durably() {
        let peers = vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)];
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("state");
        let node = kurogane_raft::Node::new(RaftNodeId(1), peers, 5, 2).expect("valid node");
        let storage = Storage::open(&path).expect("open storage");
        let mut actor = Actor::new(
            Replica::new(node),
            storage,
            RecordingTransport::new(),
            u64::MAX,
        );

        actor
            .handle_event(Event::Step {
                from: RaftNodeId(2),
                message: Message::RequestVote(kurogane_raft::RequestVote {
                    term: 1,
                    candidate_id: RaftNodeId(2),
                    last_log_index: 0,
                    last_log_term: 0,
                }),
            })
            .expect("handle event");

        assert_eq!(
            actor.transport.sent,
            vec![(
                RaftNodeId(2),
                Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: true,
                })
            )]
        );

        let reopened = Storage::open(&path).expect("reopen storage");
        assert_eq!(reopened.hard_state().current_term, 1);
        assert_eq!(reopened.hard_state().voted_for, Some(RaftNodeId(2)));
    }

    #[test]
    fn handle_peer_request_returns_the_reply_directly_without_double_sending() {
        let peers = vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)];
        let mut actor = actor(RaftNodeId(1), peers, 5, 2);

        let reply = actor
            .handle_peer_request(
                RaftNodeId(2),
                Message::RequestVote(kurogane_raft::RequestVote {
                    term: 1,
                    candidate_id: RaftNodeId(2),
                    last_log_index: 0,
                    last_log_term: 0,
                }),
            )
            .expect("handle peer request")
            .expect("recognized member, request produces a reply");

        assert_eq!(
            reply,
            Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            })
        );
        // The reply went back as this call's return value, not through
        // PeerTransport -- nothing was queued for outbound delivery.
        assert!(actor.transport.sent.is_empty());
    }

    #[test]
    fn handle_peer_request_from_a_nonmember_returns_none_without_panicking() {
        let peers = vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)];
        let mut actor = actor(RaftNodeId(1), peers, 5, 2);

        let reply = actor
            .handle_peer_request(
                RaftNodeId(9),
                Message::RequestVote(kurogane_raft::RequestVote {
                    term: 1,
                    candidate_id: RaftNodeId(9),
                    last_log_index: 0,
                    last_log_term: 0,
                }),
            )
            .expect("handle peer request");

        assert_eq!(reply, None);
        assert_eq!(actor.node().current_term(), 0);
    }

    #[tokio::test]
    async fn actor_handle_round_trips_a_peer_request_through_the_channel() {
        let peers = vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)];
        let actor = actor(RaftNodeId(1), peers, 5, 2);
        let (handle, receiver) = channel(8);
        tokio::spawn(run(actor, receiver));

        let reply = handle
            .peer_request(
                RaftNodeId(2),
                Message::RequestVote(kurogane_raft::RequestVote {
                    term: 1,
                    candidate_id: RaftNodeId(2),
                    last_log_index: 0,
                    last_log_term: 0,
                }),
            )
            .await
            .expect("actor task is alive")
            .expect("recognized member, request produces a reply");

        assert_eq!(
            reply,
            Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            })
        );
    }

    #[tokio::test]
    async fn actor_handle_round_trips_propose_on_a_single_node_cluster() {
        let actor = actor(RaftNodeId(1), vec![RaftNodeId(1)], 1, 1);
        let (handle, receiver) = channel(8);
        tokio::spawn(run(actor, receiver));

        handle.tick(5);
        // Give the spawned task a chance to process the tick before
        // proposing; propose() itself awaits a reply, so no sleep is
        // needed beyond that ordering guarantee once it's in the channel.
        let outcome = handle
            .propose(Command::Set {
                key: vec![1],
                value: vec![9],
            })
            .await
            .expect("actor task is alive");

        assert_eq!(outcome, ProposeOutcome::Accepted(1));
    }

    #[test]
    fn propose_on_a_follower_carries_a_known_leader_hint() {
        let mut actor = actor(
            RaftNodeId(1),
            vec![RaftNodeId(1), RaftNodeId(2), RaftNodeId(3)],
            5,
            2,
        );
        actor
            .handle_event(Event::Step {
                from: RaftNodeId(2),
                message: Message::AppendEntries(kurogane_raft::AppendEntries {
                    term: 1,
                    leader_id: RaftNodeId(2),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: Vec::new(),
                    leader_commit: 0,
                }),
            })
            .expect("handle event");

        let outcome = actor
            .propose(Command::Set {
                key: vec![1],
                value: vec![2],
            })
            .expect("propose");

        assert_eq!(outcome, ProposeOutcome::NotLeader(Some(RaftNodeId(2))));
    }
}
