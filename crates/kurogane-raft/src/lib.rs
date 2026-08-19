//! Transport-free types and state ownership for Kurogane's Raft core.

use std::collections::{BTreeMap, BTreeSet};
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

/// One entry in a node's replicated log. `command` is opaque bytes;
/// interpreting it as a client command is milestone four's job.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogEntry {
    pub term: u64,
    pub command: Vec<u8>,
}

/// A leader's replication message. Empty `entries` is a heartbeat.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AppendEntries {
    pub term: u64,
    pub leader_id: NodeId,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<LogEntry>,
    pub leader_commit: u64,
}

/// The response to an `AppendEntries`. `match_index` is meaningful only when
/// `success` is true: how far the responder's log now matches the leader's.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendEntriesResponse {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

/// A leader's message installing a full snapshot on a follower whose
/// `next_index` has fallen at or behind the leader's own compaction
/// boundary — sent instead of `AppendEntries` in that case, since the
/// entries it would otherwise need are gone. `data` is opaque to this
/// crate, exactly like `LogEntry.command`. Single-shot: a toy-sized KV's
/// state fits one message, and this crate does no chunking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InstallSnapshot {
    pub term: u64,
    pub leader_id: NodeId,
    pub last_included_index: u64,
    pub last_included_term: u64,
    pub data: Vec<u8>,
}

/// The response to an `InstallSnapshot`. `last_included_index` echoes back
/// the point now covered, so the leader can advance `next_index`/
/// `match_index` for this peer the same way `AppendEntriesResponse.
/// match_index` lets it do for ordinary replication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InstallSnapshotResponse {
    pub term: u64,
    pub last_included_index: u64,
}

/// A message understood by the transport-free Raft core.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Message {
    RequestVote(RequestVote),
    RequestVoteResponse(RequestVoteResponse),
    AppendEntries(AppendEntries),
    AppendEntriesResponse(AppendEntriesResponse),
    InstallSnapshot(InstallSnapshot),
    InstallSnapshotResponse(InstallSnapshotResponse),
}

/// One explicit input to a node transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Event {
    Tick { next_timeout: u64 },
    Step { from: NodeId, message: Message },
}

/// One side effect emitted by a node transition for its owner to interpret.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Effect {
    Send {
        to: NodeId,
        message: Message,
    },
    /// Must be made durable before any later effect in the same returned
    /// `Vec` is honored (in particular, any `Send` that depends on it).
    PersistHardState {
        term: u64,
        voted_for: Option<NodeId>,
    },
    /// Splice: the durable log should discard every entry at or after
    /// `from_index`, then have `entries` appended in its place. Same
    /// before-any-dependent-Send ordering guarantee as `PersistHardState`.
    PersistLog {
        from_index: u64,
        entries: Vec<LogEntry>,
    },
    /// Replaces the durable snapshot: `data` (opaque to this crate — see
    /// `LogEntry.command`'s identical treatment) becomes the new snapshot
    /// bytes, and every log entry at or before `last_included_index`
    /// becomes stale, since it's now represented by the snapshot instead.
    /// This effect alone does not fully describe post-compaction log state
    /// (a real installation may also need to retain or discard a suffix
    /// above the boundary) — it is always immediately followed by a
    /// `PersistLog` in the same batch that pins down exactly what remains.
    /// Same before-any-dependent-Send ordering guarantee as the others.
    PersistSnapshot {
        last_included_index: u64,
        last_included_term: u64,
        data: Vec<u8>,
    },
}

/// The subset of a node's state that must survive a crash: its current term
/// and which candidate (if any) it voted for in that term. Everything else —
/// `commit_index`, role, and leader-only bookkeeping like `next_index` — is
/// not persistent state and is safely re-derived after a restart.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HardState {
    pub current_term: u64,
    pub voted_for: Option<NodeId>,
}

/// Metadata about the most recent snapshot a node holds. Log indices at or
/// before `last_included_index` have been compacted away and are no longer
/// stored in `log`; `last_included_index == 0` (the default) means no
/// snapshot exists yet, and every absolute index still lives in `log`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SnapshotMetadata {
    pub last_included_index: u64,
    pub last_included_term: u64,
}

/// A snapshot boundary paired with its opaque bytes. Only exists as a
/// grouping for `Node::recover`'s parameter list (mirroring `HardState`
/// pairing term/vote) — `Node` itself keeps the two as separate fields, and
/// `snapshot()`/`snapshot_data()` return them separately too.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Snapshot {
    pub metadata: SnapshotMetadata,
    pub data: Vec<u8>,
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

/// Invalid compaction request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactError {
    /// `up_to_index` is past `commit_index` — compacting it would discard
    /// state no majority has actually agreed on yet.
    IndexNotCommitted,
}

impl fmt::Display for CompactError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::IndexNotCommitted => "cannot compact past the commit index",
        };

        formatter.write_str(message)
    }
}

impl Error for CompactError {}

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
    log: Vec<LogEntry>,
    commit_index: u64,
    next_index: BTreeMap<NodeId, u64>,
    match_index: BTreeMap<NodeId, u64>,
    snapshot: SnapshotMetadata,
    /// Opaque bytes for `snapshot`, kept around so a leader can resend them
    /// via `InstallSnapshot` without asking its owner for them again.
    snapshot_data: Vec<u8>,
}

impl Node {
    /// Constructs a follower in term zero from a fixed, canonical membership.
    /// A fresh node is just the degenerate case of having recovered from no
    /// prior hard state and an empty log.
    pub fn new(
        id: NodeId,
        peers: Vec<NodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
    ) -> Result<Self, ConfigError> {
        Self::recover(
            id,
            peers,
            election_timeout,
            heartbeat_interval,
            HardState::default(),
            Vec::new(),
            Snapshot::default(),
        )
    }

    /// Reconstructs a node from durably persisted hard state, log, and
    /// snapshot after a simulated crash. Always returns a `Follower` — role
    /// is not persistent state, so a recovered node must win a fresh
    /// election to lead again — and `commit_index` starts at
    /// `snapshot.metadata.last_included_index` (zero if there's no snapshot
    /// yet), not unconditionally zero: a snapshot can only ever have been
    /// built from already-committed data, so that much of `commit_index`
    /// *is* recoverable fact, not state to re-derive. Everything above that
    /// point is still safely re-derived through normal replication.
    pub fn recover(
        id: NodeId,
        peers: Vec<NodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
        hard_state: HardState,
        log: Vec<LogEntry>,
        snapshot: Snapshot,
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
            current_term: hard_state.current_term,
            voted_for: hard_state.voted_for,
            election_elapsed: 0,
            election_timeout,
            votes_granted: BTreeSet::new(),
            heartbeat_elapsed: 0,
            heartbeat_interval,
            log,
            commit_index: snapshot.metadata.last_included_index,
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            snapshot: snapshot.metadata,
            snapshot_data: snapshot.data,
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
        self.snapshot.last_included_index + self.log.len() as u64
    }

    pub fn last_log_term(&self) -> u64 {
        self.log
            .last()
            .map(|entry| entry.term)
            .unwrap_or(self.snapshot.last_included_term)
    }

    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    pub fn commit_index(&self) -> u64 {
        self.commit_index
    }

    pub fn snapshot(&self) -> SnapshotMetadata {
        self.snapshot
    }

    pub fn snapshot_data(&self) -> &[u8] {
        &self.snapshot_data
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
                        self.on_append_entries_response(from, response)
                    }
                    Message::InstallSnapshot(request) => {
                        if request.leader_id != from {
                            return Vec::new();
                        }
                        self.on_install_snapshot(from, request)
                    }
                    Message::InstallSnapshotResponse(response) => {
                        self.on_install_snapshot_response(from, response)
                    }
                }
            }
        }
    }

    fn is_member(&self, id: NodeId) -> bool {
        self.peers.binary_search(&id).is_ok()
    }

    /// Position of absolute `index` within `log`. Callers must ensure
    /// `index > snapshot.last_included_index` — anything at or before the
    /// snapshot boundary has been compacted away and isn't stored here.
    fn vec_index(&self, index: u64) -> usize {
        (index - self.snapshot.last_included_index - 1) as usize
    }

    /// The log entry at absolute `index`, if it's still held in `log`.
    /// `None` both for an out-of-range index and for one at or before the
    /// snapshot boundary — compacted entries are reached through
    /// `InstallSnapshot`, not this accessor.
    pub fn entry_at(&self, index: u64) -> Option<&LogEntry> {
        if index == 0 || index <= self.snapshot.last_included_index {
            return None;
        }
        self.log.get(self.vec_index(index))
    }

    fn term_at(&self, index: u64) -> Option<u64> {
        if index == 0 {
            return None;
        }
        if index == self.snapshot.last_included_index {
            return Some(self.snapshot.last_included_term);
        }
        if index < self.snapshot.last_included_index {
            return None;
        }
        self.entry_at(index).map(|entry| entry.term)
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
        self.broadcast_replication()
    }

    /// Builds the `Effect::Send` this node, as leader, owes `peer` right
    /// now: an `AppendEntries` (a heartbeat when `peer` is fully caught up,
    /// real entries otherwise) when `peer`'s `next_index` is still covered
    /// by `log`, or an `InstallSnapshot` when it isn't — the entries `peer`
    /// needs have already been compacted away.
    fn replicate_to(&self, peer: NodeId) -> Effect {
        let next_index = *self
            .next_index
            .get(&peer)
            .expect("next_index tracked for every peer while leader");

        if next_index <= self.snapshot.last_included_index {
            return Effect::Send {
                to: peer,
                message: Message::InstallSnapshot(InstallSnapshot {
                    term: self.current_term,
                    leader_id: self.id,
                    last_included_index: self.snapshot.last_included_index,
                    last_included_term: self.snapshot.last_included_term,
                    data: self.snapshot_data.clone(),
                }),
            };
        }

        let prev_log_index = next_index - 1;
        let prev_log_term = self.term_at(prev_log_index).unwrap_or(0);

        Effect::Send {
            to: peer,
            message: Message::AppendEntries(AppendEntries {
                term: self.current_term,
                leader_id: self.id,
                prev_log_index,
                prev_log_term,
                entries: self.entries_from(next_index),
                leader_commit: self.commit_index,
            }),
        }
    }

    /// Absolute log indices from `index` (inclusive) to the end, as still
    /// held in `log`. Callers must ensure `index > snapshot.last_included_index`
    /// — anything at or before the boundary must go through `InstallSnapshot`
    /// instead.
    fn entries_from(&self, index: u64) -> Vec<LogEntry> {
        self.log[self.vec_index(index)..].to_vec()
    }

    fn broadcast_replication(&self) -> Vec<Effect> {
        self.peers
            .iter()
            .copied()
            .filter(|&peer| peer != self.id)
            .map(|peer| self.replicate_to(peer))
            .collect()
    }

    /// Transitions this node to `Leader`, (re)seeding per-peer replication
    /// state, and returns the immediate replication round that establishes
    /// its authority right away rather than waiting for the next heartbeat
    /// tick. Always an `AppendEntries` round at this exact point — every
    /// peer's `next_index` is freshly seeded to `last_log_index() + 1`,
    /// strictly above the snapshot boundary — but a later back-off on
    /// conflict can still push it down to `InstallSnapshot` territory.
    fn become_leader(&mut self) -> Vec<Effect> {
        self.role = Role::Leader;
        self.heartbeat_elapsed = 0;

        let next_index = self.last_log_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for &peer in self.peers.iter().filter(|&&peer| peer != self.id) {
            self.next_index.insert(peer, next_index);
            self.match_index.insert(peer, 0);
        }

        self.broadcast_replication()
    }

    fn start_election(&mut self, next_timeout: u64) -> Vec<Effect> {
        self.role = Role::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id);
        self.votes_granted.clear();
        self.votes_granted.insert(self.id);
        self.election_elapsed = 0;
        self.election_timeout = next_timeout;

        let mut effects = vec![Effect::PersistHardState {
            term: self.current_term,
            voted_for: self.voted_for,
        }];

        if self.has_quorum() {
            effects.extend(self.become_leader());
            return effects;
        }

        effects.extend(
            self.peers
                .iter()
                .copied()
                .filter(|&peer| peer != self.id)
                .map(|peer| Effect::Send {
                    to: peer,
                    message: Message::RequestVote(RequestVote {
                        term: self.current_term,
                        candidate_id: self.id,
                        last_log_index: self.last_log_index(),
                        last_log_term: self.last_log_term(),
                    }),
                }),
        );
        effects
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

        let stepped_down = request.term > self.current_term;
        if stepped_down {
            self.step_down(request.term);
        }

        let log_is_up_to_date = (request.last_log_term, request.last_log_index)
            >= (self.last_log_term(), self.last_log_index());
        let already_voted_for_candidate = self.voted_for == Some(request.candidate_id);
        let can_grant =
            log_is_up_to_date && (self.voted_for.is_none() || already_voted_for_candidate);

        if can_grant {
            self.voted_for = Some(request.candidate_id);
            self.election_elapsed = 0;
        }

        let mut effects = Vec::new();
        // Persist whenever term or vote changed, even on a rejection: a
        // step-down that clears the prior vote must survive a crash just as
        // much as a granted one does.
        if stepped_down || can_grant {
            effects.push(Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            });
        }
        effects.push(Effect::Send {
            to: from,
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: self.current_term,
                granted: can_grant,
            }),
        });
        effects
    }

    fn on_request_vote_response(
        &mut self,
        from: NodeId,
        response: RequestVoteResponse,
    ) -> Vec<Effect> {
        if response.term > self.current_term {
            self.step_down(response.term);
            return vec![Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            }];
        }

        if response.term < self.current_term || self.role != Role::Candidate {
            return Vec::new();
        }

        if response.granted {
            self.votes_granted.insert(from);
            if self.has_quorum() {
                return self.become_leader();
            }
        }

        Vec::new()
    }

    fn reject_append_entries(&self, from: NodeId) -> Effect {
        Effect::Send {
            to: from,
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: self.current_term,
                success: false,
                match_index: 0,
            }),
        }
    }

    fn on_append_entries(&mut self, from: NodeId, request: AppendEntries) -> Vec<Effect> {
        if request.term < self.current_term {
            return vec![self.reject_append_entries(from)];
        }

        let stepped_down = request.term > self.current_term;
        if stepped_down {
            self.step_down(request.term);
        } else if self.role == Role::Candidate {
            self.role = Role::Follower;
        }

        self.election_elapsed = 0;

        // A `prev_log_index` at or before our own snapshot boundary is
        // trivially satisfied: our snapshot can only ever have been built
        // from already-committed data, and Leader Completeness guarantees
        // any valid leader's log matches ours at every committed index —
        // there's nothing left to check. This is reachable in ordinary
        // operation, not just a defensive edge case: an ex-leader that
        // compacted while leading, then stepped down, can receive a new
        // leader's optimistic `next_index` guess that lands below its own
        // compacted boundary via the normal back-off-by-one retry.
        let prev_log_satisfied = request.prev_log_index <= self.snapshot.last_included_index
            || self.term_at(request.prev_log_index) == Some(request.prev_log_term);
        if !prev_log_satisfied {
            let mut effects = Vec::new();
            if stepped_down {
                effects.push(Effect::PersistHardState {
                    term: self.current_term,
                    voted_for: self.voted_for,
                });
            }
            effects.push(self.reject_append_entries(from));
            return effects;
        }

        let mut log_changed_from: Option<u64> = None;
        for (offset, entry) in request.entries.iter().enumerate() {
            let index = request.prev_log_index + offset as u64 + 1;
            if index <= self.snapshot.last_included_index {
                // Already compacted away, and guaranteed identical to what
                // the leader is sending (same reasoning as above) — nothing
                // to do.
                continue;
            }
            match self.term_at(index) {
                Some(existing_term) if existing_term == entry.term => {}
                Some(_) => {
                    self.log.truncate(self.vec_index(index));
                    self.log.push(entry.clone());
                    if log_changed_from.is_none() {
                        log_changed_from = Some(index);
                    }
                }
                None => {
                    self.log.push(entry.clone());
                    if log_changed_from.is_none() {
                        log_changed_from = Some(index);
                    }
                }
            }
        }

        if request.leader_commit > self.commit_index {
            self.commit_index = request.leader_commit.min(self.last_log_index());
        }

        let match_index = request.prev_log_index + request.entries.len() as u64;

        let mut effects = Vec::new();
        if stepped_down {
            effects.push(Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            });
        }
        if let Some(from_index) = log_changed_from {
            effects.push(Effect::PersistLog {
                from_index,
                entries: self.entries_from(from_index),
            });
        }
        effects.push(Effect::Send {
            to: from,
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: self.current_term,
                success: true,
                match_index,
            }),
        });
        effects
    }

    fn install_snapshot_response(&self, last_included_index: u64) -> InstallSnapshotResponse {
        InstallSnapshotResponse {
            term: self.current_term,
            last_included_index,
        }
    }

    /// Handles an incoming `InstallSnapshot` from a recognized leader. Every
    /// path here ends in exactly one `Send` — `Actor::handle_peer_request`
    /// (the runtime layer) pops the last effect as the direct RPC reply and
    /// panics if it isn't one.
    fn on_install_snapshot(&mut self, from: NodeId, request: InstallSnapshot) -> Vec<Effect> {
        if request.term < self.current_term {
            return vec![Effect::Send {
                to: from,
                message: Message::InstallSnapshotResponse(
                    self.install_snapshot_response(self.snapshot.last_included_index),
                ),
            }];
        }

        let stepped_down = request.term > self.current_term;
        if stepped_down {
            self.step_down(request.term);
        } else if self.role == Role::Candidate {
            self.role = Role::Follower;
        }

        self.election_elapsed = 0;

        let mut effects = Vec::new();
        if stepped_down {
            effects.push(Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            });
        }

        // §7 retain rule: if we already have this exact point (and maybe
        // more) via ordinary replication -- whether or not we've compacted
        // it into our own snapshot yet -- this install is stale or
        // duplicate. No-op: don't touch the log or the snapshot, just
        // acknowledge what we already have.
        let already_covered =
            self.term_at(request.last_included_index) == Some(request.last_included_term);
        if !already_covered {
            self.log.clear();
            self.snapshot = SnapshotMetadata {
                last_included_index: request.last_included_index,
                last_included_term: request.last_included_term,
            };
            self.snapshot_data = request.data.clone();
            self.commit_index = self.commit_index.max(request.last_included_index);

            effects.push(Effect::PersistSnapshot {
                last_included_index: request.last_included_index,
                last_included_term: request.last_included_term,
                data: request.data,
            });
            // PersistSnapshot alone doesn't fully describe post-install log
            // state -- pin down that nothing survives above the new
            // boundary (any prior conflicting suffix in durable storage is
            // now stale) with an explicit empty splice.
            effects.push(Effect::PersistLog {
                from_index: request.last_included_index + 1,
                entries: Vec::new(),
            });
        }

        effects.push(Effect::Send {
            to: from,
            message: Message::InstallSnapshotResponse(
                self.install_snapshot_response(request.last_included_index),
            ),
        });
        effects
    }

    fn on_install_snapshot_response(
        &mut self,
        from: NodeId,
        response: InstallSnapshotResponse,
    ) -> Vec<Effect> {
        if response.term > self.current_term {
            self.step_down(response.term);
            return vec![Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            }];
        }

        if response.term < self.current_term || self.role != Role::Leader {
            return Vec::new();
        }

        let match_index = *self
            .match_index
            .get(&from)
            .expect("match_index tracked for every peer while leader");
        let match_index = match_index.max(response.last_included_index);
        self.match_index.insert(from, match_index);
        self.next_index.insert(from, match_index + 1);
        self.advance_commit_index();

        let next_index = *self
            .next_index
            .get(&from)
            .expect("next_index tracked for every peer while leader");
        if next_index <= self.last_log_index() {
            vec![self.replicate_to(from)]
        } else {
            Vec::new()
        }
    }

    fn on_append_entries_response(
        &mut self,
        from: NodeId,
        response: AppendEntriesResponse,
    ) -> Vec<Effect> {
        if response.term > self.current_term {
            self.step_down(response.term);
            return vec![Effect::PersistHardState {
                term: self.current_term,
                voted_for: self.voted_for,
            }];
        }

        if response.term < self.current_term || self.role != Role::Leader {
            return Vec::new();
        }

        if response.success {
            let match_index = *self
                .match_index
                .get(&from)
                .expect("match_index tracked for every peer while leader");
            let match_index = match_index.max(response.match_index);
            self.match_index.insert(from, match_index);
            self.next_index.insert(from, match_index + 1);
            self.advance_commit_index();
        } else {
            let next_index = *self
                .next_index
                .get(&from)
                .expect("next_index tracked for every peer while leader");
            self.next_index
                .insert(from, next_index.saturating_sub(1).max(1));
        }

        let next_index = *self
            .next_index
            .get(&from)
            .expect("next_index tracked for every peer while leader");
        if next_index <= self.last_log_index() {
            vec![self.replicate_to(from)]
        } else {
            Vec::new()
        }
    }

    /// The majority-commit rule restricted to the current term (Raft's
    /// Figure 8 guard): a prior-term entry never commits by count alone, only
    /// indirectly once a current-term entry at or after it does.
    fn advance_commit_index(&mut self) {
        let mut match_indices: Vec<u64> = self
            .peers
            .iter()
            .copied()
            .map(|peer| {
                if peer == self.id {
                    self.last_log_index()
                } else {
                    *self
                        .match_index
                        .get(&peer)
                        .expect("match_index tracked for every peer while leader")
                }
            })
            .collect();
        match_indices.sort_unstable();

        let candidate = match_indices[self.peers.len() - self.quorum_size()];
        if candidate > self.commit_index && self.term_at(candidate) == Some(self.current_term) {
            self.commit_index = candidate;
        }
    }

    /// Appends `command` to this node's log if it is the leader, returning
    /// the entry's new 1-based index and the `Effect`s needed to make it
    /// durable. Replication to peers rides the existing periodic
    /// `AppendEntries` cycle rather than sending immediately.
    pub fn propose(&mut self, command: Vec<u8>) -> Option<(u64, Vec<Effect>)> {
        if self.role != Role::Leader {
            return None;
        }

        let entry = LogEntry {
            term: self.current_term,
            command,
        };
        self.log.push(entry.clone());
        let index = self.last_log_index();
        self.advance_commit_index();

        Some((
            index,
            vec![Effect::PersistLog {
                from_index: index,
                entries: vec![entry],
            }],
        ))
    }

    /// Compacts every log entry up to and including `up_to_index` into a
    /// snapshot, discarding them from `log`. `snapshot_data` is opaque to
    /// this crate, exactly like `LogEntry.command` — producing and
    /// interpreting it is the state machine owner's job. Any role may call
    /// this, not just the leader: self-compaction is how every node bounds
    /// its own log growth, independent of who's currently leading.
    ///
    /// A no-op (`Ok(Vec::new())`) if `up_to_index` is at or before the
    /// current snapshot boundary already — the same idempotent-retry
    /// treatment every other effect-producing transition here gives a
    /// request it's already satisfied. An error if `up_to_index` reaches
    /// past `commit_index`: compacting an uncommitted entry could discard
    /// state no majority has actually agreed on yet.
    pub fn compact(
        &mut self,
        up_to_index: u64,
        snapshot_data: Vec<u8>,
    ) -> Result<Vec<Effect>, CompactError> {
        if up_to_index <= self.snapshot.last_included_index {
            return Ok(Vec::new());
        }
        if up_to_index > self.commit_index {
            return Err(CompactError::IndexNotCommitted);
        }

        let last_included_term = self
            .term_at(up_to_index)
            .expect("a committed index is always present in the log or already the boundary");
        self.log.drain(0..=self.vec_index(up_to_index));
        self.snapshot = SnapshotMetadata {
            last_included_index: up_to_index,
            last_included_term,
        };
        self.snapshot_data = snapshot_data.clone();

        Ok(vec![
            Effect::PersistSnapshot {
                last_included_index: up_to_index,
                last_included_term,
                data: snapshot_data,
            },
            // PersistSnapshot alone doesn't tell the owner's durable log to
            // drop the now-redundant compacted prefix -- pin it down
            // explicitly, same as on_install_snapshot does. entries_from
            // reads the boundary we just updated above, so this returns
            // exactly what's left in `log` after the drain.
            Effect::PersistLog {
                from_index: up_to_index + 1,
                entries: self.entries_from(up_to_index + 1),
            },
        ])
    }

    fn step_down(&mut self, term: u64) {
        self.role = Role::Follower;
        self.current_term = term;
        self.voted_for = None;
        self.votes_granted.clear();
        self.next_index.clear();
        self.match_index.clear();
    }

    fn quorum_size(&self) -> usize {
        self.peers.len() / 2 + 1
    }

    fn has_quorum(&self) -> bool {
        self.votes_granted.len() >= self.quorum_size()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        AppendEntries, AppendEntriesResponse, CompactError, ConfigError, Effect, Event, HardState,
        InstallSnapshot, InstallSnapshotResponse, LogEntry, Message, Node, NodeId, RequestVote,
        RequestVoteResponse, Role, Snapshot, SnapshotMetadata,
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
        assert_eq!(node.commit_index(), 0);
        assert!(node.log().is_empty());
        assert!(node.votes_granted().is_empty());
    }

    #[test]
    fn recover_always_yields_a_follower_with_the_given_hard_state_and_log() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let hard_state = HardState {
            current_term: 5,
            voted_for: Some(NodeId(2)),
        };
        let log = vec![LogEntry {
            term: 3,
            command: vec![1],
        }];

        let node = Node::recover(
            NodeId(1),
            peers,
            1,
            1,
            hard_state,
            log.clone(),
            Snapshot::default(),
        )
        .expect("valid node");

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 5);
        assert_eq!(node.voted_for(), Some(NodeId(2)));
        assert_eq!(node.log(), log.as_slice());
        assert_eq!(node.commit_index(), 0);
    }

    #[test]
    fn recover_initializes_commit_index_from_the_snapshot_boundary_not_zero() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let snapshot = SnapshotMetadata {
            last_included_index: 4,
            last_included_term: 2,
        };
        let log = vec![LogEntry {
            term: 2,
            command: vec![9],
        }];

        let node = Node::recover(
            NodeId(1),
            peers,
            1,
            1,
            HardState::default(),
            log,
            Snapshot {
                metadata: snapshot,
                data: vec![7, 7, 7],
            },
        )
        .expect("valid node");

        // A snapshot can only ever have been built from already-committed
        // data, so that much of commit_index is recoverable fact, not
        // state that must be re-derived from zero like role/next_index are.
        assert_eq!(node.commit_index(), 4);
        assert_eq!(node.snapshot(), snapshot);
        assert_eq!(node.snapshot_data(), &[7, 7, 7]);
        assert_eq!(node.last_log_index(), 5);
    }

    #[test]
    fn recover_validates_configuration_the_same_way_as_new() {
        let result = Node::recover(
            NodeId(1),
            vec![],
            1,
            1,
            HardState::default(),
            Vec::new(),
            Snapshot::default(),
        );

        assert_eq!(
            result.expect_err("configuration must fail"),
            ConfigError::EmptyConfiguration
        );
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
                Effect::PersistHardState {
                    term: 1,
                    voted_for: Some(NodeId(1)),
                },
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

        assert_eq!(
            effects,
            vec![Effect::PersistHardState {
                term: 1,
                voted_for: Some(NodeId(1)),
            }]
        );
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
            vec![
                Effect::PersistHardState {
                    term: 1,
                    voted_for: Some(NodeId(2)),
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 1,
                        granted: true,
                    }),
                }
            ]
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
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: Some(NodeId(3)),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 2,
                        granted: true,
                    }),
                }
            ]
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
        let expected_heartbeat = AppendEntries {
            term: 1,
            leader_id: NodeId(1),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: Vec::new(),
            leader_commit: 0,
        };
        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntries(expected_heartbeat.clone()),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::AppendEntries(expected_heartbeat),
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
        let expected_heartbeat = AppendEntries {
            term: 1,
            leader_id: NodeId(1),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: Vec::new(),
            leader_commit: 0,
        };
        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntries(expected_heartbeat.clone()),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::AppendEntries(expected_heartbeat),
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
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 1,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 1,
                        success: true,
                        match_index: 0,
                    }),
                }
            ]
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
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.election_elapsed(), 0);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: 1,
                    success: true,
                    match_index: 0,
                }),
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
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 1);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: 1,
                    success: false,
                    match_index: 0,
                }),
            }]
        );
    }

    #[test]
    fn rejects_append_entries_when_prev_log_does_not_match() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 3,
                prev_log_term: 1,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert!(node.log().is_empty());
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 1,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 1,
                        success: false,
                        match_index: 0,
                    }),
                }
            ]
        );
    }

    #[test]
    fn appends_new_entries_when_prev_log_matches() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let entries = vec![
            LogEntry {
                term: 1,
                command: Vec::new(),
            },
            LogEntry {
                term: 1,
                command: Vec::new(),
            },
        ];

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: entries.clone(),
                leader_commit: 0,
            }),
        });

        assert_eq!(node.log(), entries.as_slice());
        assert_eq!(node.last_log_index(), 2);
        assert_eq!(node.last_log_term(), 1);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 1,
                    voted_for: None,
                },
                Effect::PersistLog {
                    from_index: 1,
                    entries: entries.clone(),
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 1,
                        success: true,
                        match_index: 2,
                    }),
                }
            ]
        );
    }

    #[test]
    fn truncates_conflicting_suffix_and_appends_leaders_entries() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![
                    LogEntry {
                        term: 1,
                        command: vec![1],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![2],
                    },
                ],
                leader_commit: 0,
            }),
        });
        assert_eq!(node.last_log_index(), 2);

        let replacement = LogEntry {
            term: 2,
            command: vec![9],
        };
        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 2,
                leader_id: NodeId(2),
                prev_log_index: 1,
                prev_log_term: 1,
                entries: vec![replacement.clone()],
                leader_commit: 0,
            }),
        });

        assert_eq!(node.log().len(), 2);
        assert_eq!(node.log()[1], replacement);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: None,
                },
                Effect::PersistLog {
                    from_index: 2,
                    entries: vec![replacement],
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 2,
                        success: true,
                        match_index: 2,
                    }),
                }
            ]
        );
    }

    #[test]
    fn advances_commit_index_from_leader_commit_on_a_heartbeat() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    command: Vec::new(),
                }],
                leader_commit: 0,
            }),
        });
        assert_eq!(node.commit_index(), 0);

        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 1,
                prev_log_term: 1,
                entries: Vec::new(),
                leader_commit: 1,
            }),
        });

        assert_eq!(node.commit_index(), 1);
    }

    #[test]
    fn delivering_the_same_append_entries_twice_is_idempotent() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let request = AppendEntries {
            term: 1,
            leader_id: NodeId(2),
            prev_log_index: 0,
            prev_log_term: 0,
            entries: vec![LogEntry {
                term: 1,
                command: vec![7],
            }],
            leader_commit: 0,
        };

        let first = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(request.clone()),
        });
        let second = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(request),
        });

        assert_eq!(node.log().len(), 1);

        // The two deliveries are not effect-for-effect identical: the first
        // legitimately persists (new term, a new entry); the second changes
        // nothing and persists nothing. What "idempotent" means here is that
        // the response is the same either way, not that the two calls do
        // identical work.
        let response = |effects: &[Effect]| {
            effects.iter().find_map(|effect| match effect {
                Effect::Send { message, .. } => Some(message.clone()),
                _ => None,
            })
        };
        assert_eq!(response(&first), response(&second));
        assert_eq!(
            response(&second),
            Some(Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 1,
            }))
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
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 2);
        assert_eq!(node.voted_for(), None);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(3),
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
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 4,
                success: false,
                match_index: 0,
            }),
        });

        assert_eq!(
            effects,
            vec![Effect::PersistHardState {
                term: 4,
                voted_for: None,
            }]
        );
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
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.role(), Role::Follower);
    }

    #[test]
    fn on_append_entries_treats_prev_log_index_at_the_snapshot_boundary_as_satisfied() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![
                    LogEntry {
                        term: 1,
                        command: vec![1],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![2],
                    },
                ],
                leader_commit: 2,
            }),
        });
        assert_eq!(node.commit_index(), 2);
        node.compact(2, vec![9]).expect("index 2 is committed");
        assert!(node.log().is_empty());

        // prev_log_index sits exactly at the boundary -- term_at(2) only
        // exists via the snapshot now, not the log, but the check must
        // still pass.
        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 2,
                prev_log_term: 1,
                entries: vec![LogEntry {
                    term: 1,
                    command: vec![3],
                }],
                leader_commit: 3,
            }),
        });

        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                command: vec![3]
            }]
        );
        assert_eq!(node.last_log_index(), 3);
        assert_eq!(
            effects,
            vec![
                Effect::PersistLog {
                    from_index: 3,
                    entries: vec![LogEntry {
                        term: 1,
                        command: vec![3]
                    }],
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 1,
                        success: true,
                        match_index: 3,
                    }),
                },
            ]
        );
    }

    #[test]
    fn on_append_entries_skips_entries_at_or_before_the_snapshot_boundary() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let seed_entries = vec![
            LogEntry {
                term: 1,
                command: vec![1],
            },
            LogEntry {
                term: 1,
                command: vec![2],
            },
            LogEntry {
                term: 1,
                command: vec![3],
            },
        ];
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: seed_entries.clone(),
                leader_commit: 3,
            }),
        });
        node.compact(2, vec![9]).expect("index 2 is committed");
        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                command: vec![3]
            }]
        );

        // A new leader's optimistic next_index guess sends prev_log_index
        // 0 -- it has no idea this node has already compacted through
        // index 2. Entries at or before the boundary must be silently
        // skipped (they're guaranteed identical, already committed), not
        // rejected or reapplied.
        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::AppendEntries(AppendEntries {
                term: 2,
                leader_id: NodeId(3),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: seed_entries,
                leader_commit: 3,
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 2);
        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                command: vec![3]
            }]
        );
        assert_eq!(node.commit_index(), 3);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::AppendEntriesResponse(AppendEntriesResponse {
                        term: 2,
                        success: true,
                        match_index: 3,
                    }),
                },
            ]
        );
    }

    #[test]
    fn rejects_stale_term_install_snapshot() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.current_term(), 1);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 0,
                leader_id: NodeId(2),
                last_included_index: 5,
                last_included_term: 0,
                data: vec![9],
            }),
        });

        assert_eq!(node.role(), Role::Candidate);
        assert_eq!(node.current_term(), 1);
        assert!(node.log().is_empty());
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                    term: 1,
                    last_included_index: 0,
                }),
            }]
        );
    }

    #[test]
    fn ignores_install_snapshot_whose_leader_id_does_not_match_the_sender() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 1,
                leader_id: NodeId(3),
                last_included_index: 5,
                last_included_term: 1,
                data: vec![9],
            }),
        });

        assert!(effects.is_empty());
        assert_eq!(node.current_term(), 0);
    }

    #[test]
    fn on_install_snapshot_installs_and_discards_any_existing_log() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    command: vec![1],
                }],
                leader_commit: 0,
            }),
        });
        assert_eq!(node.log().len(), 1);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 1,
                leader_id: NodeId(2),
                last_included_index: 5,
                last_included_term: 1,
                data: vec![9, 9],
            }),
        });

        assert_eq!(
            node.snapshot(),
            SnapshotMetadata {
                last_included_index: 5,
                last_included_term: 1,
            }
        );
        assert_eq!(node.snapshot_data(), &[9, 9]);
        assert!(node.log().is_empty());
        assert_eq!(node.commit_index(), 5);
        assert_eq!(node.last_log_index(), 5);
        assert_eq!(
            effects,
            vec![
                Effect::PersistSnapshot {
                    last_included_index: 5,
                    last_included_term: 1,
                    data: vec![9, 9],
                },
                Effect::PersistLog {
                    from_index: 6,
                    entries: Vec::new(),
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                        term: 1,
                        last_included_index: 5,
                    }),
                },
            ]
        );
    }

    #[test]
    fn on_install_snapshot_steps_down_on_higher_term() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Candidate);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 2,
                leader_id: NodeId(2),
                last_included_index: 3,
                last_included_term: 2,
                data: vec![1],
            }),
        });

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 2);
        assert_eq!(node.voted_for(), None);
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 2,
                    voted_for: None,
                },
                Effect::PersistSnapshot {
                    last_included_index: 3,
                    last_included_term: 2,
                    data: vec![1],
                },
                Effect::PersistLog {
                    from_index: 4,
                    entries: Vec::new(),
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                        term: 2,
                        last_included_index: 3,
                    }),
                },
            ]
        );
    }

    #[test]
    fn on_install_snapshot_is_a_no_op_when_the_receiver_already_covers_that_point() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![
                    LogEntry {
                        term: 1,
                        command: vec![1],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![2],
                    },
                ],
                leader_commit: 2,
            }),
        });
        assert_eq!(node.log().len(), 2);
        assert_eq!(node.commit_index(), 2);

        // A stale/duplicate InstallSnapshot for a point already covered by
        // ordinary replication -- no-op, keep the existing (further-ahead)
        // log rather than blowing it away.
        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 1,
                leader_id: NodeId(2),
                last_included_index: 1,
                last_included_term: 1,
                data: vec![9, 9],
            }),
        });

        assert_eq!(node.log().len(), 2);
        assert_eq!(node.snapshot(), SnapshotMetadata::default());
        assert!(node.snapshot_data().is_empty());
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                    term: 1,
                    last_included_index: 1,
                }),
            }]
        );
    }

    #[test]
    fn grants_vote_when_candidate_log_is_at_least_as_up_to_date() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let seed_entries = vec![1, 1, 1, 2]
            .into_iter()
            .map(|term| LogEntry {
                term,
                command: Vec::new(),
            })
            .collect::<Vec<_>>();
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: seed_entries,
                leader_commit: 0,
            }),
        });
        assert_eq!(node.last_log_index(), 4);
        assert_eq!(node.last_log_term(), 2);

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
            vec![
                Effect::PersistHardState {
                    term: 1,
                    voted_for: Some(NodeId(2)),
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 1,
                        granted: true,
                    }),
                }
            ]
        );
    }

    #[test]
    fn rejects_vote_when_candidate_log_is_behind_even_after_stepping_down_on_a_higher_term() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");
        let seed_entries = vec![1, 2, 3, 3]
            .into_iter()
            .map(|term| LogEntry {
                term,
                command: Vec::new(),
            })
            .collect::<Vec<_>>();
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 3,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: seed_entries,
                leader_commit: 0,
            }),
        });
        assert_eq!(node.last_log_index(), 4);
        assert_eq!(node.last_log_term(), 3);

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
            vec![
                // The step-down cleared the term-3 vote; that must persist
                // even though this candidate's stale log means no new vote
                // is granted.
                Effect::PersistHardState {
                    term: 9,
                    voted_for: None,
                },
                Effect::Send {
                    to: NodeId(2),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 9,
                        granted: false,
                    }),
                }
            ]
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

        assert_eq!(
            effects,
            vec![Effect::PersistHardState {
                term: 5,
                voted_for: None,
            }]
        );
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

    /// A 3-node leader (node 1) with an established quorum (node 2 voted for
    /// it), ready for propose/replication tests.
    fn established_leader() -> Node {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 1 });
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 1,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);
        node
    }

    #[test]
    fn propose_returns_none_when_not_leader() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let result = node.propose(vec![1, 2, 3]);

        assert_eq!(result, None);
        assert!(node.log().is_empty());
    }

    #[test]
    fn propose_appends_entry_and_returns_index_for_leader() {
        let mut node = established_leader();

        let (index, effects) = node.propose(vec![9]).expect("leader accepts propose");

        assert_eq!(index, 1);
        assert_eq!(
            effects,
            vec![Effect::PersistLog {
                from_index: 1,
                entries: vec![LogEntry {
                    term: 1,
                    command: vec![9],
                }],
            }]
        );
        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                command: vec![9],
            }]
        );
    }

    #[test]
    fn propose_commits_immediately_in_a_single_node_cluster() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Leader);

        let (index, _effects) = node.propose(vec![1]).expect("leader accepts propose");

        assert_eq!(index, 1);
        assert_eq!(node.commit_index(), 1);
    }

    #[test]
    fn compact_discards_the_committed_prefix_and_persists_the_snapshot() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.propose(vec![1]);
        node.propose(vec![2]);
        node.propose(vec![3]);
        assert_eq!(node.commit_index(), 3);

        let effects = node.compact(2, vec![9, 9]).expect("2 is committed");

        assert_eq!(
            effects,
            vec![
                Effect::PersistSnapshot {
                    last_included_index: 2,
                    last_included_term: 1,
                    data: vec![9, 9],
                },
                Effect::PersistLog {
                    from_index: 3,
                    entries: vec![LogEntry {
                        term: 1,
                        command: vec![3],
                    }],
                },
            ]
        );
        assert_eq!(
            node.snapshot(),
            SnapshotMetadata {
                last_included_index: 2,
                last_included_term: 1,
            }
        );
        assert_eq!(node.snapshot_data(), &[9, 9]);
        // Compacted-away entries are gone from log(), but absolute indexing
        // (last_log_index, entry_at) is unaffected by the boundary shift.
        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                command: vec![3]
            }]
        );
        assert_eq!(node.last_log_index(), 3);
        assert_eq!(node.entry_at(2), None);
        assert_eq!(
            node.entry_at(3),
            Some(&LogEntry {
                term: 1,
                command: vec![3]
            })
        );
    }

    #[test]
    fn compact_through_the_entire_log_persists_an_empty_retained_suffix() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.propose(vec![1]);
        node.propose(vec![2]);

        let effects = node.compact(2, vec![9]).expect("2 is committed");

        assert_eq!(
            effects,
            vec![
                Effect::PersistSnapshot {
                    last_included_index: 2,
                    last_included_term: 1,
                    data: vec![9],
                },
                Effect::PersistLog {
                    from_index: 3,
                    entries: Vec::new(),
                },
            ]
        );
        assert!(node.log().is_empty());
    }

    #[test]
    fn compact_past_commit_index_is_rejected() {
        let mut node = established_leader();
        node.propose(vec![1]);
        assert_eq!(node.commit_index(), 0);

        let error = node
            .compact(1, Vec::new())
            .expect_err("index 1 is not committed yet");

        assert_eq!(error, CompactError::IndexNotCommitted);
    }

    #[test]
    fn compact_at_or_before_the_current_boundary_is_a_no_op() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.propose(vec![1]);
        node.propose(vec![2]);
        node.compact(1, vec![1]).expect("1 is committed");

        let effects = node
            .compact(1, vec![9])
            .expect("already-compacted index is a no-op");

        assert!(effects.is_empty());
        assert_eq!(
            node.snapshot(),
            SnapshotMetadata {
                last_included_index: 1,
                last_included_term: 1,
            }
        );
        assert_eq!(node.snapshot_data(), &[1]);
    }

    #[test]
    fn on_install_snapshot_response_advances_match_and_next_index_and_retries() {
        let mut node = established_leader();
        node.propose(vec![1]);
        node.propose(vec![2]);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                term: 1,
                last_included_index: 1,
            }),
        });

        assert_eq!(node.commit_index(), 1);
        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntries(AppendEntries {
                    term: 1,
                    leader_id: NodeId(1),
                    prev_log_index: 1,
                    prev_log_term: 1,
                    entries: vec![LogEntry {
                        term: 1,
                        command: vec![2],
                    }],
                    leader_commit: 1,
                }),
            }]
        );
    }

    #[test]
    fn on_install_snapshot_response_steps_down_on_higher_term() {
        let mut node = established_leader();

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::InstallSnapshotResponse(InstallSnapshotResponse {
                term: 5,
                last_included_index: 0,
            }),
        });

        assert_eq!(
            effects,
            vec![Effect::PersistHardState {
                term: 5,
                voted_for: None,
            }]
        );
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 5);
    }

    #[test]
    fn on_append_entries_response_advances_match_and_next_index_on_success() {
        let mut node = established_leader();
        node.propose(vec![1]);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 1,
            }),
        });

        assert_eq!(node.commit_index(), 1);
        assert!(effects.is_empty());
    }

    #[test]
    fn on_append_entries_response_backs_off_next_index_and_retries_on_failure() {
        let mut node = established_leader();
        node.propose(vec![1]);
        node.propose(vec![2]);

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: false,
                match_index: 0,
            }),
        });

        assert_eq!(
            effects,
            vec![Effect::Send {
                to: NodeId(2),
                message: Message::AppendEntries(AppendEntries {
                    term: 1,
                    leader_id: NodeId(1),
                    prev_log_index: 0,
                    prev_log_term: 0,
                    entries: vec![
                        LogEntry {
                            term: 1,
                            command: vec![1],
                        },
                        LogEntry {
                            term: 1,
                            command: vec![2],
                        },
                    ],
                    leader_commit: 0,
                }),
            }]
        );
    }

    #[test]
    fn leader_sends_install_snapshot_instead_of_append_entries_once_a_peers_next_index_falls_to_or_below_the_boundary()
     {
        let mut node = established_leader();
        node.propose(vec![1]);
        node.propose(vec![2]);
        node.propose(vec![3]);

        // node2 acks all three entries; node3 never has -- its next_index
        // stays at the initial seed of 1 from becoming leader with an
        // empty log.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 3,
            }),
        });
        assert_eq!(node.commit_index(), 3);

        node.compact(3, vec![9, 9]).expect("index 3 is committed");

        let effects = node.step(Event::Tick { next_timeout: 5 });

        assert_eq!(
            effects,
            vec![
                Effect::Send {
                    to: NodeId(2),
                    message: Message::AppendEntries(AppendEntries {
                        term: 1,
                        leader_id: NodeId(1),
                        prev_log_index: 3,
                        prev_log_term: 1,
                        entries: Vec::new(),
                        leader_commit: 3,
                    }),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::InstallSnapshot(InstallSnapshot {
                        term: 1,
                        leader_id: NodeId(1),
                        last_included_index: 3,
                        last_included_term: 1,
                        data: vec![9, 9],
                    }),
                },
            ]
        );
    }

    #[test]
    fn leader_does_not_commit_a_prior_term_entry_by_replica_count_alone() {
        let mut node = established_leader();
        node.propose(vec![1]); // entry A: term 1, index 1
        assert_eq!(node.commit_index(), 0);

        // Node 1 is deposed by a higher-term leader but keeps entry A in its
        // log, then wins a later election itself.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 2,
                leader_id: NodeId(2),
                prev_log_index: 1,
                prev_log_term: 1,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.log().len(), 1);

        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.current_term(), 3);
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: 3,
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        node.propose(vec![2]); // entry B: term 3, index 2

        // A majority (self + node 2) now has entry A, but not the current
        // term's entry B. Entry A must not commit by count alone.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 3,
                success: true,
                match_index: 1,
            }),
        });
        assert_eq!(node.commit_index(), 0);

        // Once a majority also has entry B, this node's own current-term
        // entry, both entries commit together.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 3,
                success: true,
                match_index: 2,
            }),
        });
        assert_eq!(node.commit_index(), 2);
    }
}
