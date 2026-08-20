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

/// A cluster's voting membership. `voters` is the active (or, during a
/// joint-consensus transition, the *new*) set; `old_voters` is `Some` only
/// while a transition is in flight, holding the set being replaced. Both
/// sets must independently reach majority for anything — an election or a
/// commit — to succeed while `old_voters` is `Some`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClusterConfig {
    pub voters: Vec<NodeId>,
    pub old_voters: Option<Vec<NodeId>>,
}

impl ClusterConfig {
    /// Whether `id` is a voter under either set this config carries.
    pub fn is_voter(&self, id: NodeId) -> bool {
        self.voters.contains(&id) || self.old_voters.as_deref().unwrap_or(&[]).contains(&id)
    }

    /// Whether this config is mid-transition (`C_old,new`, in the paper's
    /// notation) rather than stable.
    pub fn is_joint(&self) -> bool {
        self.old_voters.is_some()
    }
}

/// A log entry's payload. `Command` bytes are opaque to this crate — see
/// `LogEntry`'s own doc comment — but `Configuration` is not: cluster
/// membership is core protocol state, not an application concern, so this
/// crate parses and acts on a `Configuration` entry immediately, even
/// before it commits, exactly as it does for every other piece of Figure-2
/// state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LogPayload {
    Command(Vec<u8>),
    Configuration(ClusterConfig),
}

/// One entry in a node's replicated log. A `Command` payload is opaque
/// bytes; interpreting it as a client command is `kurogane-kv`'s job. A
/// `Configuration` payload is not opaque — see `LogPayload`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LogEntry {
    pub term: u64,
    pub payload: LogPayload,
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
    /// The membership active as of this snapshot's boundary, so a receiver
    /// that catches up purely through this transfer (never having seen the
    /// `Configuration` log entries it summarizes) still learns current
    /// membership — see `Effect::PersistSnapshot`'s identical field.
    pub config: ClusterConfig,
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
        /// The membership active as of this snapshot's boundary. Compaction
        /// can discard the very log entries that established the current
        /// configuration, so — exactly like `last_included_index`/
        /// `last_included_term` — the boundary must carry membership
        /// forward explicitly, or a node that recovers or catches up
        /// purely from a snapshot transfer (not ordinary log replication)
        /// has no way to learn it. Mirrors `Node`'s own in-memory
        /// `snapshot_config` field one-for-one.
        config: ClusterConfig,
    },
    /// Replaces the durable learner set with `learners`. A separate effect
    /// from `PersistSnapshot` rather than folded into it: the learner set
    /// changes independently of snapshot/compaction timing (via
    /// `add_learner`/`remove_learner`), so tying its durability to
    /// whenever a compaction happens to occur would mean a freshly added
    /// learner could be lost on a leader crash long before the next
    /// compaction ever runs. A fresh joiner's own empty storage has
    /// nothing else to tell it what it is, so this is genuinely durable
    /// state, not leader-local bookkeeping. Same before-any-dependent-Send
    /// ordering guarantee as the others.
    PersistLearners {
        learners: Vec<NodeId>,
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
    /// The membership active as of `metadata`'s boundary — see
    /// `Effect::PersistSnapshot`'s identical field for why this travels
    /// alongside the boundary rather than being re-derived.
    pub config: ClusterConfig,
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
    role: Role,
    current_term: u64,
    voted_for: Option<NodeId>,
    election_elapsed: u64,
    election_timeout: u64,
    votes_granted: BTreeSet<NodeId>,
    heartbeat_elapsed: u64,
    heartbeat_interval: u64,
    /// Ticks elapsed since this node last accepted an `AppendEntries` or
    /// `InstallSnapshot` from a current-or-higher-term leader -- reset only
    /// in those two handlers' success paths, alongside `election_elapsed`.
    /// Deliberately NOT reset by granting a vote or by starting an
    /// election, unlike `election_elapsed`: it exists specifically so
    /// `on_request_vote`'s Follower/Candidate disruption guard has a
    /// "heard from a leader recently" signal that vote-granting can't
    /// collide with (`election_elapsed` can't serve that role -- see the
    /// guard's doc comment). Transient timing state, same category as
    /// `election_elapsed`/`heartbeat_elapsed`: not part of `HardState`, not
    /// persisted, safely re-initialized on `recover`/`new` -- to
    /// `heartbeat_interval`, not 0, since 0 would misrepresent a node that
    /// has never heard from any leader as having just done so.
    leader_contact_elapsed: u64,
    log: Vec<LogEntry>,
    commit_index: u64,
    next_index: BTreeMap<NodeId, u64>,
    match_index: BTreeMap<NodeId, u64>,
    snapshot: SnapshotMetadata,
    /// Opaque bytes for `snapshot`, kept around so a leader can resend them
    /// via `InstallSnapshot` without asking its owner for them again.
    snapshot_data: Vec<u8>,
    /// The configuration active as of `snapshot`'s boundary — the fallback
    /// `current_config` resolves to once no `Configuration` entry remains
    /// in `log` (either because none was ever appended, or because
    /// `compact` drained the one that was).
    snapshot_config: ClusterConfig,
    /// The configuration currently governing quorum and voting eligibility:
    /// the most recent `Configuration` entry found scanning `log`
    /// backward, or `snapshot_config` if none remains. Live the instant a
    /// `Configuration` entry is appended, even before it commits — Raft's
    /// membership rule, not an oversight — so this is recomputed after
    /// every mutation that can change what "latest" means (`propose`'s
    /// append, `on_append_entries`'s merge/truncate, and `compact`'s
    /// drain), never read lazily.
    current_config: ClusterConfig,
    /// The absolute log index of the entry that established `current_config`
    /// -- `snapshot.last_included_index` if `current_config` came from
    /// `snapshot_config` rather than a live log entry. Kept in lockstep with
    /// `current_config` by `recompute_config` (and `recover`'s equivalent
    /// initialization); `advance_commit_index` compares `commit_index`
    /// against this to know when the currently active configuration entry
    /// itself has committed, which is what triggers the automatic
    /// `C_old,new` -> `C_new` follow-up and the not-in-`C_new` leader
    /// step-down.
    current_config_index: u64,
    /// Non-voting members this node (while leader) replicates to but never
    /// counts toward any quorum -- how a new server joins with an empty
    /// log without being able to disrupt the cluster (§6). Durably
    /// persisted via `Effect::PersistLearners`, independent of
    /// `current_config`/`snapshot_config`: promotion to a voter is a
    /// separate, explicit `propose_config_change` call, not implied by
    /// membership here.
    learners: BTreeSet<NodeId>,
}

/// The most recent `Configuration` payload in `log`, scanning backward,
/// paired with its absolute log index -- `snapshot_last_included_index` is
/// the boundary `log[0]` sits just above, so `log[i]`'s absolute index is
/// `snapshot_last_included_index + i + 1`. `None` if every entry (if any) is
/// a `Command`.
fn latest_configuration_in(
    log: &[LogEntry],
    snapshot_last_included_index: u64,
) -> Option<(ClusterConfig, u64)> {
    log.iter()
        .enumerate()
        .rev()
        .find_map(|(offset, entry)| match &entry.payload {
            LogPayload::Configuration(config) => Some((
                config.clone(),
                snapshot_last_included_index + offset as u64 + 1,
            )),
            LogPayload::Command(_) => None,
        })
}

/// Whether `reached` holds for a strict majority of `set`, independent of
/// any other set — the building block for joint consensus's dual-majority
/// rule, which requires this to hold separately for both the old and new
/// voter sets while a configuration transition is in flight.
fn majority_reached(set: &[NodeId], reached: impl Fn(NodeId) -> bool) -> bool {
    let quorum = set.len() / 2 + 1;
    set.iter().filter(|&&id| reached(id)).count() >= quorum
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
            Vec::new(),
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
    ///
    /// `learners` joins the parameter list as durable state alongside
    /// `hard_state`/`log`/`snapshot`, not bundled into any one of them:
    /// it's persisted independently via `Effect::PersistLearners`, on its
    /// own schedule, not tied to hard-state or snapshot timing.
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors HardState/log/Snapshot's own flat parameters -- see the doc comment"
    )]
    pub fn recover(
        id: NodeId,
        peers: Vec<NodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
        hard_state: HardState,
        log: Vec<LogEntry>,
        snapshot: Snapshot,
        learners: Vec<NodeId>,
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

        // A non-default snapshot means a real prior snapshot exists, and its
        // own `config` field holds the actual durable membership as of that
        // boundary -- preferring `peers` here would silently discard
        // mid-transition membership on every ordinary restart, undoing the
        // entire "membership lives in replicated state" property this
        // milestone exists to establish. Only a genuinely fresh node (no
        // snapshot at all, `last_included_index == 0`) falls back to
        // wrapping `peers` as the bootstrap (C_0) configuration.
        let snapshot_config = if snapshot.metadata.last_included_index > 0 {
            snapshot.config.clone()
        } else {
            ClusterConfig {
                voters: peers,
                old_voters: None,
            }
        };
        let (current_config, current_config_index) =
            latest_configuration_in(&log, snapshot.metadata.last_included_index).unwrap_or_else(
                || {
                    (
                        snapshot_config.clone(),
                        snapshot.metadata.last_included_index,
                    )
                },
            );

        Ok(Self {
            id,
            role: Role::Follower,
            current_term: hard_state.current_term,
            voted_for: hard_state.voted_for,
            election_elapsed: 0,
            election_timeout,
            votes_granted: BTreeSet::new(),
            heartbeat_elapsed: 0,
            heartbeat_interval,
            // Starts at (not below) heartbeat_interval, not 0: 0 would mean
            // "just heard from a leader," which is false for a node that
            // has never heard from one -- a freshly constructed/recovered
            // node must not spuriously trip the Follower/Candidate
            // disruption guard in on_request_vote before any real leader
            // contact has happened.
            leader_contact_elapsed: heartbeat_interval,
            log,
            commit_index: snapshot.metadata.last_included_index,
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            snapshot: snapshot.metadata,
            snapshot_data: snapshot.data,
            snapshot_config,
            current_config,
            current_config_index,
            learners: learners.into_iter().collect(),
        })
    }

    /// Constructs a brand-new node joining an existing cluster for the
    /// first time: no durable state of its own yet, and — unlike
    /// `new`/`recover` — deliberately **not** a voter under any
    /// configuration, not even a degenerate one-member `[id]` config. A
    /// join-mode node with a self-inclusive bootstrap config would believe
    /// itself a one-voter cluster and start deposing the real leader the
    /// instant its election timeout first fires; going through `recover`
    /// at all is wrong here; not just inconvenient, since `recover`
    /// unconditionally requires `peers` to be non-empty and self-inclusive
    /// (`EmptyConfiguration`/`LocalNodeMissing`) even when the snapshot
    /// path would otherwise discard it. This node is safe to sit idle
    /// indefinitely: `start_election`'s campaign-eligibility invariant
    /// (never campaign unless `self.id` is a voter under `current_config`)
    /// already refuses to let an empty-config node run for election, so it
    /// waits passively until a real leader's `AppendEntries`/
    /// `InstallSnapshot` admits it — first as a learner (an existing
    /// leader's `add_learner`, replicating without granting voting
    /// rights), later as a full voter once a `propose_config_change` that
    /// includes this node's id actually commits.
    pub fn new_learner(
        id: NodeId,
        election_timeout: u64,
        heartbeat_interval: u64,
    ) -> Result<Self, ConfigError> {
        if election_timeout == 0 {
            return Err(ConfigError::ZeroElectionTimeout);
        }
        if heartbeat_interval == 0 {
            return Err(ConfigError::ZeroHeartbeatInterval);
        }
        if heartbeat_interval > election_timeout {
            return Err(ConfigError::HeartbeatIntervalExceedsElectionTimeout);
        }

        let empty_config = ClusterConfig {
            voters: Vec::new(),
            old_voters: None,
        };

        Ok(Self {
            id,
            role: Role::Follower,
            current_term: 0,
            voted_for: None,
            election_elapsed: 0,
            election_timeout,
            votes_granted: BTreeSet::new(),
            heartbeat_elapsed: 0,
            heartbeat_interval,
            leader_contact_elapsed: heartbeat_interval,
            log: Vec::new(),
            commit_index: 0,
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            snapshot: SnapshotMetadata::default(),
            snapshot_data: Vec::new(),
            snapshot_config: empty_config.clone(),
            current_config: empty_config,
            current_config_index: 0,
            learners: BTreeSet::new(),
        })
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    /// The configuration currently governing quorum and voting eligibility
    /// — see the field doc comment on `Node::current_config` for exactly
    /// when this changes.
    pub fn current_config(&self) -> &ClusterConfig {
        &self.current_config
    }

    /// The active voter set under `current_config()`. A direct drop-in for
    /// callers that only ever want a flat member list (most callers, most
    /// of the time, since `old_voters` is `None` outside a joint-consensus
    /// transition) — use `current_config()` directly when the full shape,
    /// including a possible `old_voters`, actually matters.
    pub fn voters(&self) -> &[NodeId] {
        &self.current_config.voters
    }

    /// Recomputes `current_config` from scratch: the latest `Configuration`
    /// entry in `log`, or `snapshot_config` if none remains. Called after
    /// every mutation that can change what "latest" means.
    fn recompute_config(&mut self) {
        let (current_config, current_config_index) =
            latest_configuration_in(&self.log, self.snapshot.last_included_index).unwrap_or_else(
                || {
                    (
                        self.snapshot_config.clone(),
                        self.snapshot.last_included_index,
                    )
                },
            );
        self.current_config = current_config;
        self.current_config_index = current_config_index;
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

    pub fn leader_contact_elapsed(&self) -> u64 {
        self.leader_contact_elapsed
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

    /// The configuration active as of the current snapshot boundary —
    /// `current_config()`'s fallback once no `Configuration` entry remains
    /// in `log`. Most callers want `current_config()`, not this.
    pub fn snapshot_config(&self) -> &ClusterConfig {
        &self.snapshot_config
    }

    /// Non-voting members currently tracked for replication -- see the
    /// field doc comment on `Node::learners`.
    pub fn learners(&self) -> &BTreeSet<NodeId> {
        &self.learners
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

    /// Whether `id` is trusted as a sender at `step()`'s message-dispatch
    /// choke point: a voter under `current_config` (voters ∪ old_voters)
    /// or a tracked learner. The learner half matters specifically for a
    /// leader (or any node) receiving a message *from* a learner -- e.g.
    /// an `AppendEntriesResponse` from a learner catching up -- without it,
    /// that response's `from` would fail this check and get silently
    /// dropped by `step()`, permanently stalling the learner's progress
    /// tracking. (A learner's own incoming `AppendEntries` *from* its
    /// leader already passes today, since the leader is always a voter --
    /// this widening is about the reverse direction.)
    ///
    /// Third clause: a brand-new `Node::new_learner` node's own
    /// `current_config` is genuinely empty (`voters: []`, `old_voters:
    /// None`) -- by construction, the *only* path that ever produces that
    /// shape, since `Node::new`/`Node::recover` both reject an empty
    /// bootstrap set and `propose_config_change` rejects an empty
    /// `new_voters`. Without this clause, such a node's very first contact
    /// -- the leader's own `AddLearner`-triggered `AppendEntries`/
    /// `InstallSnapshot` -- would itself fail this check (it's judged
    /// against *this* node's config, not the sender's), permanently
    /// deadlocking the join: the node can never learn who its leader is,
    /// because it refuses to listen to anyone before it already knows. The
    /// real trust boundary for this first contact is the runtime's shared
    /// cluster-token authentication, not this predicate -- once a real
    /// `AppendEntries`/`InstallSnapshot` lands and installs a real
    /// configuration, `current_config.voters` is no longer empty and this
    /// bypass naturally stops applying.
    fn is_member(&self, id: NodeId) -> bool {
        self.current_config.is_voter(id)
            || self.learners.contains(&id)
            || (self.current_config.voters.is_empty() && self.current_config.old_voters.is_none())
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
        self.leader_contact_elapsed += 1;
        if self.election_elapsed < self.election_timeout {
            return Vec::new();
        }

        self.start_election(next_timeout)
    }

    fn on_leader_tick(&mut self) -> Vec<Effect> {
        self.heartbeat_elapsed += 1;
        self.leader_contact_elapsed += 1;
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
                    config: self.snapshot_config.clone(),
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

    /// The union of every peer this node, as leader, owes ordinary
    /// replication to: `current_config.voters` ∪ `current_config.old_voters`
    /// ∪ `learners`, minus itself. A `BTreeSet` naturally dedupes a peer
    /// that's a member of more than one of those sets at once (e.g.
    /// present in both `voters` and `old_voters` mid-transition, or --
    /// less likely, but defensive against caller misuse -- both `learners`
    /// and a voter set). `self.id` is filtered defensively even though it
    /// should never legitimately appear in `learners`.
    fn replication_targets(&self) -> BTreeSet<NodeId> {
        self.current_config
            .voters
            .iter()
            .copied()
            .chain(self.current_config.old_voters.iter().flatten().copied())
            .chain(self.learners.iter().copied())
            .filter(|&peer| peer != self.id)
            .collect()
    }

    fn broadcast_replication(&self) -> Vec<Effect> {
        self.replication_targets()
            .into_iter()
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
    /// Seeds over `replication_targets()` (voters ∪ old_voters ∪
    /// learners), not just `current_config.voters` -- an old-set-only
    /// voter or a learner that responds before the first ordinary
    /// `broadcast_replication` sweep still needs a tracked entry to update.
    fn become_leader(&mut self) -> Vec<Effect> {
        self.role = Role::Leader;
        self.heartbeat_elapsed = 0;

        let next_index = self.last_log_index() + 1;
        self.next_index.clear();
        self.match_index.clear();
        for peer in self.replication_targets() {
            self.next_index.insert(peer, next_index);
            self.match_index.insert(peer, 0);
        }

        self.broadcast_replication()
    }

    /// Handles `Event::Tick`'s election-timeout path. A true no-op for this
    /// tick -- no term bump, no `Candidate` transition -- unless `self.id`
    /// is currently a voter (voters ∪ old_voters) under `current_config`.
    /// This single invariant is what prevents three different disruption
    /// scenarios: a learner self-electing before it's ever promoted, a
    /// leader (or any server) just excluded from a committed `C_new`
    /// re-electing itself, and a config-less joining node self-electing
    /// (that last scenario needs a join-mode constructor this stage
    /// doesn't add, but the invariant itself is general, not special-cased
    /// to any one of the three).
    fn start_election(&mut self, next_timeout: u64) -> Vec<Effect> {
        if !self.current_config.is_voter(self.id) {
            return Vec::new();
        }

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
            self.current_config
                .voters
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

        // Removed-server disruption guard (paper §6 / dissertation §4.2.3):
        // a node that IS the leader right now unconditionally disregards
        // any incoming RequestVote -- including one with a higher term --
        // without stepping down. `on_tick` returns early for
        // `Role::Leader` (see `on_leader_tick`), so this node's own
        // `election_elapsed` is frozen and isn't a meaningful "heard from
        // a leader recently" signal the way it is for a Follower/
        // Candidate; being the leader is itself the strongest possible
        // instance of that signal. A genuinely stale or partitioned
        // leader still learns of the new term normally once the partition
        // heals, via AppendEntriesResponse/InstallSnapshotResponse term
        // comparison -- not via RequestVote.
        if self.role == Role::Leader {
            return vec![Effect::Send {
                to: from,
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: self.current_term,
                    granted: false,
                }),
            }];
        }

        // The paper's matching Follower/Candidate-side guard: disregard an
        // incoming RequestVote -- including a higher-term one -- without
        // stepping down or granting, while this node has heard from a
        // current-or-higher-term leader within the last `heartbeat_interval`
        // ticks. This is deliberately keyed on `leader_contact_elapsed`, not
        // `election_elapsed`: `election_elapsed` is also reset by granting a
        // vote (below) and by `start_election`, so for any voting Follower/
        // Candidate, `election_elapsed < election_timeout` holds as an
        // invariant essentially all the time -- it becomes false only by
        // the node itself becoming a fresh Candidate first, which
        // immediately resets it back to 0. Keying the guard on that
        // comparison would make ordinary vote-granting permanently
        // unreachable for every voter, breaking essentially every
        // multi-node election in the test suite -- confirmed empirically,
        // not merely reasoned about. `leader_contact_elapsed` has no such
        // collision: it is reset only by `on_append_entries`/
        // `on_install_snapshot`'s success paths, never by granting a vote
        // or starting an election, so it tracks leader contact and nothing
        // else.
        //
        // Deliberately scoped to `current_config.old_voters.is_some()` --
        // an active joint-consensus transition -- rather than applying
        // unconditionally. `step()`'s `is_member` choke point already drops
        // a RequestVote from any node fully outside `current_config`
        // (neither a voter, an old voter, nor a learner), which is the
        // *only* way a genuinely removed server's message could reach this
        // function at all; the sole remaining window for the paper's
        // concern is a server still tracked in `old_voters` during the
        // joint phase. Applying the guard unconditionally instead would
        // catch nothing that scoping it this way doesn't already catch,
        // while regressing ordinary (non-transitioning) election liveness:
        // confirmed empirically -- an unconditional version delays a
        // legitimate re-election by up to one `heartbeat_interval` for any
        // follower that received the old (now genuinely dead/partitioned)
        // leader's last heartbeat moments before the partition, breaking
        // pre-existing election tests that predate this milestone and have
        // every right to assume it doesn't change their behavior.
        // `self.role` is Follower or Candidate here -- Leader already
        // returned above.
        if self.current_config.old_voters.is_some()
            && self.leader_contact_elapsed < self.heartbeat_interval
        {
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
        self.leader_contact_elapsed = 0;

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
        if log_changed_from.is_some() {
            self.recompute_config();
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
        self.leader_contact_elapsed = 0;

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
            self.snapshot_config = request.config.clone();
            self.commit_index = self.commit_index.max(request.last_included_index);
            // The log this snapshot summarizes is gone (cleared above), so
            // recompute_config's fallback chain now resolves to exactly
            // the snapshot_config just installed -- unless a genuinely
            // more recent Configuration entry somehow survives above the
            // new boundary, which recompute_config's existing logic
            // already handles correctly by construction.
            self.recompute_config();

            effects.push(Effect::PersistSnapshot {
                last_included_index: request.last_included_index,
                last_included_term: request.last_included_term,
                data: request.data,
                config: request.config,
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
        let mut effects = self.advance_commit_index();

        // advance_commit_index may have just retired this node from
        // leadership (config-triggered step-down) -- a node that just
        // demoted itself to Follower must not still emit leader
        // replication traffic in this same batch, and next_index/
        // match_index are no longer tracked to look `from` up in anyway.
        if self.role == Role::Leader {
            let next_index = *self
                .next_index
                .get(&from)
                .expect("next_index tracked for every peer while leader");
            if next_index <= self.last_log_index() {
                effects.push(self.replicate_to(from));
            }
        }
        effects
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

        let mut effects = Vec::new();
        if response.success {
            let match_index = *self
                .match_index
                .get(&from)
                .expect("match_index tracked for every peer while leader");
            let match_index = match_index.max(response.match_index);
            self.match_index.insert(from, match_index);
            self.next_index.insert(from, match_index + 1);
            effects = self.advance_commit_index();
        } else {
            let next_index = *self
                .next_index
                .get(&from)
                .expect("next_index tracked for every peer while leader");
            self.next_index
                .insert(from, next_index.saturating_sub(1).max(1));
        }

        // advance_commit_index may have just retired this node from
        // leadership (config-triggered step-down) -- a node that just
        // demoted itself to Follower must not still emit leader
        // replication traffic in this same batch, and next_index/
        // match_index are no longer tracked to look `from` up in anyway.
        if self.role == Role::Leader {
            let next_index = *self
                .next_index
                .get(&from)
                .expect("next_index tracked for every peer while leader");
            if next_index <= self.last_log_index() {
                effects.push(self.replicate_to(from));
            }
        }
        effects
    }

    /// The majority-commit rule restricted to the current term (Raft's
    /// Figure 8 guard): a prior-term entry never commits by count alone, only
    /// indirectly once a current-term entry at or after it does.
    ///
    /// While `current_config.old_voters` is `Some` (a joint-consensus
    /// transition is in flight), an entry only commits once it's matched by
    /// a majority of *both* the new voters and the old voters — the
    /// paper's dual-majority rule. The combined candidate is the minimum of
    /// the two sets' own thresholds, not the maximum: taking the maximum
    /// would let an entry commit on one side's majority alone, exactly the
    /// split-brain scenario joint consensus exists to prevent.
    ///
    /// Leader-only side effects, checked in order once `commit_index` has
    /// been updated above, each re-reading `current_config`/
    /// `current_config_index` fresh at the point it runs rather than a
    /// value cached before this call: a single call can never satisfy both
    /// in one pass, because the first check's own append (if it fires)
    /// changes `current_config` before the second check would even look at
    /// it, and the second check's condition (a *plain* config's commit)
    /// can't become true in the same call that just produced a *plain*
    /// config from a joint one — that new entry starts uncommitted.
    ///
    /// - If `commit_index` just reached or passed the index of a still-
    ///   joint `current_config`, this leader appends the automatic
    ///   `C_old,new` -> `C_new` follow-up entry (same voters, no
    ///   `old_voters`) to its own log — the first place in this codebase a
    ///   function produces a new log entry as a side effect of commit
    ///   advancement rather than an explicit caller request.
    /// - If `commit_index` just reached or passed the index of a still-
    ///   plain `current_config` that excludes `self.id`, this leader
    ///   retires via `retire_from_leadership`.
    fn advance_commit_index(&mut self) -> Vec<Effect> {
        let mut new_match_indices: Vec<u64> = self
            .current_config
            .voters
            .iter()
            .copied()
            .map(|peer| {
                if peer == self.id {
                    self.last_log_index()
                } else {
                    // A voter `propose_config_change` just introduced may
                    // still be unseeded in match_index -- become_leader
                    // and add_learner both seed replication_targets()
                    // (voters ∪ old_voters ∪ learners) at the point they
                    // run, but a brand-new NodeId that was never a voter/
                    // old-voter and never went through add_learner first
                    // is still untracked here. Same treatment as the
                    // old-set branch below: an untracked peer counts as
                    // "not caught up" (index 0), not assumed progress.
                    self.match_index.get(&peer).copied().unwrap_or(0)
                }
            })
            .collect();
        new_match_indices.sort_unstable();
        let new_candidate =
            new_match_indices[self.current_config.voters.len() - self.quorum_size()];

        let candidate = match &self.current_config.old_voters {
            None => new_candidate,
            Some(old_voters) => {
                // A peer present only in `old_voters` (already removed
                // from `voters` by this transition) is normally still
                // seeded -- become_leader/add_learner both cover
                // old_voters -- but a peer removed before this leader's
                // own election never was. Treat an untracked peer as "not
                // caught up" (index 0) rather than assuming progress.
                let mut old_match_indices: Vec<u64> = old_voters
                    .iter()
                    .copied()
                    .map(|peer| {
                        if peer == self.id {
                            self.last_log_index()
                        } else {
                            self.match_index.get(&peer).copied().unwrap_or(0)
                        }
                    })
                    .collect();
                old_match_indices.sort_unstable();
                let old_quorum = old_voters.len() / 2 + 1;
                let old_candidate = old_match_indices[old_voters.len() - old_quorum];
                new_candidate.min(old_candidate)
            }
        };

        if candidate > self.commit_index && self.term_at(candidate) == Some(self.current_term) {
            self.commit_index = candidate;
        }

        let mut effects = Vec::new();

        if self.current_config.old_voters.is_some()
            && self.commit_index >= self.current_config_index
        {
            let follow_up = LogEntry {
                term: self.current_term,
                payload: LogPayload::Configuration(ClusterConfig {
                    voters: self.current_config.voters.clone(),
                    old_voters: None,
                }),
            };
            self.log.push(follow_up.clone());
            self.recompute_config();
            let index = self.last_log_index();
            effects.push(Effect::PersistLog {
                from_index: index,
                entries: vec![follow_up],
            });
            // Mirrors propose()'s own shape: a fresh append needs its own
            // commit-advancement pass too (the single-node-cluster
            // immediate-commit case lands here), which may itself chain
            // into the step-down check below.
            effects.extend(self.advance_commit_index());
        }

        // If the branch above just recursed, that inner call may already
        // have retired this node (its own copy of this same check, run on
        // the post-append state before returning up the stack) -- this
        // re-check then simply repeats the same true/false verdict against
        // now-identical state. Deliberately still a plain `if`, not
        // `else if`: the two checks must each read live state at the point
        // they run, and collapsing them would silently stop re-checking
        // after a non-recursing joint commit that didn't itself qualify.
        if self.current_config.old_voters.is_none()
            && self.commit_index >= self.current_config_index
            && !self.current_config.voters.contains(&self.id)
        {
            self.retire_from_leadership();
        }

        effects
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
            payload: LogPayload::Command(command),
        };
        self.log.push(entry.clone());
        self.recompute_config();
        let index = self.last_log_index();

        let mut effects = vec![Effect::PersistLog {
            from_index: index,
            entries: vec![entry],
        }];
        effects.extend(self.advance_commit_index());

        Some((index, effects))
    }

    /// Leader-only: begins a joint-consensus configuration change to
    /// `new_voters`. Mirrors `propose`'s exact shape — appends a
    /// `Configuration` entry (here, the `C_old,new` joint config: `voters:
    /// new_voters`, `old_voters: Some(current_config.voters.clone())`),
    /// takes effect immediately (even before it commits, Raft's membership
    /// rule), and returns the new entry's index plus the effects needed to
    /// make it durable. `None` if not leader, exactly like `propose`.
    ///
    /// Everything past this one entry — the automatic `C_new` follow-up
    /// once the joint entry commits, and this leader's own step-down if
    /// `C_new` excludes it once *that* commits — is driven entirely by
    /// `advance_commit_index`, not by any second call here.
    ///
    /// Rejects an empty `new_voters` with `None`: unlike `propose`'s opaque
    /// `command` bytes, an empty voter set is a real invariant violation
    /// here — `quorum_size`/`advance_commit_index`'s majority arithmetic
    /// indexes `voters.len() - quorum_size()`, which underflows for a
    /// zero-length set.
    ///
    /// A genuinely new `NodeId` in `new_voters` is always safe for
    /// commit-index arithmetic, seeded or not — an unseeded peer counts as
    /// "not caught up" (index 0) rather than panicking. Replication is a
    /// separate concern: `become_leader` and `add_learner` both seed
    /// `next_index`/`match_index` over `replication_targets()` (voters ∪
    /// old_voters ∪ learners) at the point *they* run, but this call does
    /// not re-seed anything itself. Introducing a `NodeId` this call has
    /// never seen before — one that was never a voter/old-voter as of the
    /// last election and was never added via `add_learner` — still panics
    /// in `replicate_to`'s `.expect(...)` on the next `broadcast_replication`
    /// sweep. `add_learner` first, then promoting via this call once
    /// caught up, is the safe on-ramp for a genuinely new server; promoting
    /// an already-known voter/old-voter (the common case: adding or
    /// removing among peers already seeded) is safe either way. Any
    /// learner named in `new_voters` is dropped from `learners` here, so
    /// the two sets never overlap after a promotion.
    pub fn propose_config_change(&mut self, new_voters: Vec<NodeId>) -> Option<(u64, Vec<Effect>)> {
        if self.role != Role::Leader || new_voters.is_empty() {
            return None;
        }

        // Any learner named in the new voter set is being promoted, not
        // merely still tracked -- drop it from `learners` so the two sets
        // don't overlap post-promotion (voter status now comes entirely
        // from `current_config`, the same way it already does for every
        // other voter). Persisted like any other change to `learners`,
        // ahead of the dependent Sends `advance_commit_index` may add
        // below, same before-any-dependent-Send ordering every other
        // effect in this crate follows.
        let learner_count_before = self.learners.len();
        self.learners.retain(|id| !new_voters.contains(id));
        let mut effects = Vec::new();
        if self.learners.len() != learner_count_before {
            effects.push(Effect::PersistLearners {
                learners: self.learners.iter().copied().collect(),
            });
        }

        let entry = LogEntry {
            term: self.current_term,
            payload: LogPayload::Configuration(ClusterConfig {
                voters: new_voters,
                old_voters: Some(self.current_config.voters.clone()),
            }),
        };
        self.log.push(entry.clone());
        self.recompute_config();
        let index = self.last_log_index();

        effects.push(Effect::PersistLog {
            from_index: index,
            entries: vec![entry],
        });
        effects.extend(self.advance_commit_index());

        Some((index, effects))
    }

    /// Leader-only: starts tracking `id` as a non-voting learner, so it
    /// begins receiving ordinary replication (`become_leader`/
    /// `broadcast_replication` both iterate `replication_targets()`,
    /// which includes `learners`) without being counted toward any
    /// quorum. This is the safe on-ramp for a genuinely new server: add it
    /// as a learner, let it catch up via ordinary replication, then
    /// promote it with `propose_config_change` once its `match_index` is
    /// close to the leader's — `propose_config_change` itself never seeds
    /// a brand-new `NodeId`, so promoting one that was never a learner
    /// first still panics the next heartbeat sweep (see its own doc
    /// comment).
    ///
    /// A no-op (empty effects, `self.learners` untouched) if this node
    /// isn't the leader, if `id` is already tracked, or if `id` is
    /// already a voter under `current_config` — a voter doesn't need
    /// learner tracking, and letting the two states coexist would be a
    /// confusing dual membership. Otherwise seeds `next_index`/
    /// `match_index` for `id` exactly like `become_leader` would have had
    /// `id` been a member at election time (`.entry(...).or_insert(...)`,
    /// so an id that already has tracked progress from an earlier
    /// tenure — e.g. a former voter — keeps it rather than resetting to
    /// 0), then emits `Effect::PersistLearners` with the full updated set.
    pub fn add_learner(&mut self, id: NodeId) -> Vec<Effect> {
        if self.role != Role::Leader
            || self.learners.contains(&id)
            || self.current_config.is_voter(id)
        {
            return Vec::new();
        }

        self.learners.insert(id);
        let next_index = self.last_log_index() + 1;
        self.next_index.entry(id).or_insert(next_index);
        self.match_index.entry(id).or_insert(0);

        vec![Effect::PersistLearners {
            learners: self.learners.iter().copied().collect(),
        }]
    }

    /// Leader-only: stops tracking `id` as a learner (e.g. because
    /// `propose_config_change` just promoted it to a voter, or it's being
    /// removed from the cluster entirely). A no-op (empty effects,
    /// `self.learners` untouched) if this node isn't the leader or `id`
    /// isn't currently tracked. Deliberately leaves `next_index`/
    /// `match_index` untouched: `id` may since have been promoted and
    /// still be a voter this leader is actively replicating to, and
    /// clearing its seeded progress here would panic the next
    /// `broadcast_replication` sweep exactly like never seeding it at all
    /// would.
    pub fn remove_learner(&mut self, id: NodeId) -> Vec<Effect> {
        if self.role != Role::Leader || !self.learners.contains(&id) {
            return Vec::new();
        }

        self.learners.remove(&id);

        vec![Effect::PersistLearners {
            learners: self.learners.iter().copied().collect(),
        }]
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
        // The config as of the new boundary is whatever the latest
        // Configuration entry *within the drained prefix* says, not simply
        // current_config as it stands right now: current_config may be
        // established by an entry above up_to_index (config takes effect
        // immediately, even before commit -- and up_to_index can be no
        // higher than commit_index), which survives this drain but is
        // still eligible to be truncated away later by a conflicting
        // AppendEntries. If that later truncation happens, recompute_config
        // falls back to snapshot_config -- which must hold the config that
        // was actually true at this boundary, not one from an entry this
        // compaction never touched and that turned out not to survive. If
        // no Configuration entry falls within the drained prefix, the
        // config at the new boundary is unchanged from the config at the
        // old boundary, so snapshot_config is left exactly as it was.
        let drained_prefix = &self.log[..=self.vec_index(up_to_index)];
        if let Some((config, _)) =
            latest_configuration_in(drained_prefix, self.snapshot.last_included_index)
        {
            self.snapshot_config = config;
        }
        self.log.drain(0..=self.vec_index(up_to_index));
        self.snapshot = SnapshotMetadata {
            last_included_index: up_to_index,
            last_included_term,
        };
        self.snapshot_data = snapshot_data.clone();
        self.recompute_config();

        Ok(vec![
            Effect::PersistSnapshot {
                last_included_index: up_to_index,
                last_included_term,
                data: snapshot_data,
                config: self.snapshot_config.clone(),
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

    /// Demotes a leader that has just discovered its own exclusion from a
    /// newly-committed plain (non-joint) configuration. Distinct from
    /// `step_down`: this isn't triggered by discovering a higher term, so
    /// `current_term`/`voted_for` are left untouched — only role and the
    /// same leader-only bookkeeping `step_down` clears (`votes_granted`/
    /// `next_index`/`match_index`, all safely re-derivable) reset.
    fn retire_from_leadership(&mut self) {
        self.role = Role::Follower;
        self.votes_granted.clear();
        self.next_index.clear();
        self.match_index.clear();
    }

    fn quorum_size(&self) -> usize {
        self.current_config.voters.len() / 2 + 1
    }

    /// Whether an election has been won. While `current_config.old_voters`
    /// is `Some`, this requires a majority of the old voters too, not just
    /// the new ones — the same dual-majority rule `advance_commit_index`
    /// applies to commitment, applied here to elections instead.
    fn has_quorum(&self) -> bool {
        let new_majority = self.votes_granted.len() >= self.quorum_size();
        match &self.current_config.old_voters {
            None => new_majority,
            Some(old_voters) => {
                new_majority && majority_reached(old_voters, |id| self.votes_granted.contains(&id))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{
        AppendEntries, AppendEntriesResponse, ClusterConfig, CompactError, ConfigError, Effect,
        Event, HardState, InstallSnapshot, InstallSnapshotResponse, LogEntry, LogPayload, Message,
        Node, NodeId, RequestVote, RequestVoteResponse, Role, Snapshot, SnapshotMetadata,
    };

    #[test]
    fn constructs_initial_follower_state() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let node = Node::new(NodeId(2), peers.clone(), 11, 4).expect("valid configuration");

        assert_eq!(node.id(), NodeId(2));
        assert_eq!(node.voters(), peers);
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: peers.clone(),
                old_voters: None,
            }
        );
        assert_eq!(
            node.snapshot_config(),
            &ClusterConfig {
                voters: peers,
                old_voters: None,
            }
        );
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
    fn new_learner_constructs_a_node_that_is_a_voter_under_no_configuration() {
        let node = Node::new_learner(NodeId(4), 5, 1).expect("valid join-mode node");

        assert_eq!(node.id(), NodeId(4));
        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 0);
        assert_eq!(node.voted_for(), None);
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: Vec::new(),
                old_voters: None,
            }
        );
        assert!(node.voters().is_empty());
        assert!(node.log().is_empty());
        assert_eq!(node.commit_index(), 0);
    }

    #[test]
    fn new_learner_never_self_elects_even_after_many_ticks() {
        let mut node = Node::new_learner(NodeId(4), 3, 1).expect("valid join-mode node");

        for _ in 0..10 {
            let effects = node.step(Event::Tick { next_timeout: 3 });
            assert!(
                effects.is_empty(),
                "a node that is a voter under no configuration must never campaign"
            );
        }

        assert_eq!(node.role(), Role::Follower);
        assert_eq!(node.current_term(), 0);
    }

    #[test]
    fn new_learner_rejects_the_same_degenerate_timeouts_as_new() {
        assert_eq!(
            Node::new_learner(NodeId(1), 0, 1).unwrap_err(),
            ConfigError::ZeroElectionTimeout
        );
        assert_eq!(
            Node::new_learner(NodeId(1), 1, 0).unwrap_err(),
            ConfigError::ZeroHeartbeatInterval
        );
        assert_eq!(
            Node::new_learner(NodeId(1), 1, 2).unwrap_err(),
            ConfigError::HeartbeatIntervalExceedsElectionTimeout
        );
    }

    #[test]
    fn a_join_mode_node_accepts_its_first_append_entries_from_an_unrecognized_sender() {
        // A fresh Node::new_learner has an empty current_config -- is_member
        // judges the *sender* against this node's own config, so without a
        // bypass for the genuinely-unconfigured case, this node would
        // reject the very first AppendEntries that's supposed to admit it,
        // deadlocking the join permanently (it can never learn who its
        // leader is, because it refuses to listen to anyone first).
        let mut node = Node::new_learner(NodeId(4), 5, 1).expect("valid join-mode node");

        let effects = node.step(Event::Step {
            from: NodeId(1),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(1),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    payload: LogPayload::Configuration(ClusterConfig {
                        voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                        old_voters: None,
                    }),
                }],
                leader_commit: 0,
            }),
        });

        assert!(
            effects.iter().any(|effect| matches!(
                effect,
                Effect::Send {
                    message: Message::AppendEntriesResponse(response),
                    ..
                } if response.success
            )),
            "the join-mode node must accept a real leader's first AppendEntries, not silently \
             drop it via is_member, got {effects:?}"
        );
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                old_voters: None,
            },
            "the AppendEntries carried a Configuration entry admitting this node's cluster -- \
             current_config must reflect it immediately"
        );
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
            payload: LogPayload::Command(vec![1]),
        }];

        let node = Node::recover(
            NodeId(1),
            peers,
            1,
            1,
            hard_state,
            log.clone(),
            Snapshot::default(),
            Vec::new(),
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
            payload: LogPayload::Command(vec![9]),
        }];

        let node = Node::recover(
            NodeId(1),
            peers.clone(),
            1,
            1,
            HardState::default(),
            log,
            Snapshot {
                metadata: snapshot,
                data: vec![7, 7, 7],
                // A real (non-default) snapshot with a plausible config, not
                // ClusterConfig::default() -- a genuinely empty voter set
                // here would still pass this test's own assertions (none of
                // which touch current_config), but leaves a voterless node
                // that panics the moment anything calls advance_commit_index
                // on it, which is a confusing trap for a fixture to set for
                // future readers.
                config: ClusterConfig {
                    voters: peers,
                    old_voters: None,
                },
            },
            Vec::new(),
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
    fn recover_prefers_the_snapshots_own_config_over_the_peers_argument() {
        // The peers argument names a stale/irrelevant set on purpose -- a
        // real snapshot's own `config` field is the durable membership that
        // must win, since it may reflect a mid-transition config the peers
        // argument (e.g. an operator's KUROGANE_PEERS) has no way to know
        // about. If recover regresses to wrapping `peers` again, this test
        // fails because current_config() would come back as [1, 2, 3, 9]
        // instead of the snapshot's [1, 2, 3, 4].
        let peers = vec![NodeId(1), NodeId(2), NodeId(3), NodeId(9)];
        let snapshot_voters = vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)];

        let node = Node::recover(
            NodeId(1),
            peers,
            1,
            1,
            HardState::default(),
            Vec::new(),
            Snapshot {
                metadata: SnapshotMetadata {
                    last_included_index: 4,
                    last_included_term: 2,
                },
                data: vec![1, 2, 3],
                config: ClusterConfig {
                    voters: snapshot_voters.clone(),
                    old_voters: None,
                },
            },
            Vec::new(),
        )
        .expect("valid node");

        assert_eq!(node.current_config().voters, snapshot_voters);
        assert_eq!(node.snapshot_config().voters, snapshot_voters);
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
            Vec::new(),
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

    /// A leader disregards an incoming RequestVote -- even one with a
    /// higher term -- unconditionally, without stepping down: the
    /// removed-server disruption guard's Role::Leader special case
    /// (`on_tick` returns early for a leader, so its own election_elapsed
    /// is frozen and not a meaningful "heard from a leader recently"
    /// signal -- being the leader is itself the strongest such signal).
    /// Regardless of how many heartbeat ticks pass in between, since those
    /// ticks never touch election_elapsed for a leader either. This
    /// supersedes this codebase's pre-guard behavior, where a leader used
    /// to step down and grant on any higher-term RequestVote like an
    /// ordinary follower would.
    #[test]
    fn leader_disregards_a_higher_term_request_vote_without_stepping_down() {
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

        // A few heartbeat ticks pass -- on_leader_tick never advances
        // election_elapsed, so this must not matter to the guard.
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 2,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(
            node.role(),
            Role::Leader,
            "a leader must not step down on an incoming RequestVote, however high its term"
        );
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.voted_for(), Some(NodeId(1)));
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
    fn a_freshly_constructed_node_does_not_spuriously_trigger_the_disruption_guard() {
        let node =
            Node::new(NodeId(1), vec![NodeId(1), NodeId(2), NodeId(3)], 5, 2).expect("valid node");
        assert_eq!(
            node.leader_contact_elapsed(),
            2,
            "a node that has never heard from any leader must not read as \
             though it just did -- it starts at heartbeat_interval, not 0"
        );
    }

    #[test]
    fn a_follower_in_a_joint_transition_disregards_a_higher_term_request_vote_while_leader_contact_is_recent()
     {
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(4), NodeId(5)],
            old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
        };
        let log = vec![LogEntry {
            term: 0,
            payload: LogPayload::Configuration(joint_config),
        }];
        let mut node = Node::recover(
            NodeId(2),
            vec![NodeId(1), NodeId(2), NodeId(3)],
            5,
            1,
            HardState::default(),
            log,
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");

        // A legitimate heartbeat from the current leader resets
        // leader_contact_elapsed to 0.
        node.step(Event::Step {
            from: NodeId(1),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(1),
                prev_log_index: 1,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });
        assert_eq!(node.leader_contact_elapsed(), 0);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 5,
                candidate_id: NodeId(3),
                last_log_index: 1,
                last_log_term: 0,
            }),
        });

        assert_eq!(
            node.current_term(),
            1,
            "a stray higher-term RequestVote must not move the term while \
             leader contact is recent and a joint transition is in flight"
        );
        assert_eq!(node.voted_for(), None);
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
    fn a_follower_in_a_joint_transition_grants_a_higher_term_request_vote_once_leader_contact_is_stale()
     {
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(4), NodeId(5)],
            old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
        };
        let log = vec![LogEntry {
            term: 0,
            payload: LogPayload::Configuration(joint_config),
        }];
        let mut node = Node::recover(
            NodeId(2),
            vec![NodeId(1), NodeId(2), NodeId(3)],
            5,
            1,
            HardState::default(),
            log,
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");

        node.step(Event::Step {
            from: NodeId(1),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(1),
                prev_log_index: 1,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });
        assert_eq!(node.leader_contact_elapsed(), 0);

        // One tick clears leader_contact_elapsed past heartbeat_interval
        // (1): the guard must fall through to ordinary voting rules.
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.leader_contact_elapsed(), 1);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 5,
                candidate_id: NodeId(3),
                last_log_index: 1,
                last_log_term: 0,
            }),
        });

        assert_eq!(
            node.current_term(),
            5,
            "once leader contact is stale, an ordinary higher-term \
             RequestVote is handled normally, joint transition or not"
        );
        assert_eq!(node.voted_for(), Some(NodeId(3)));
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 5,
                    voted_for: Some(NodeId(3)),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 5,
                        granted: true,
                    }),
                }
            ]
        );
    }

    #[test]
    fn the_disruption_guard_does_not_apply_outside_a_joint_consensus_transition() {
        // Same recent-leader-contact setup as the joint-transition guard
        // test, but with a single, stable (non-joint) configuration --
        // step()'s is_member choke point already drops a RequestVote from
        // any node fully outside current_config, which is the only way a
        // genuinely removed server's message could reach this function at
        // all, so the guard has nothing left to protect against here and
        // must not fire.
        let mut node =
            Node::new(NodeId(2), vec![NodeId(1), NodeId(2), NodeId(3)], 5, 1).expect("valid node");

        node.step(Event::Step {
            from: NodeId(1),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(1),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: Vec::new(),
                leader_commit: 0,
            }),
        });
        assert_eq!(node.leader_contact_elapsed(), 0);

        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 5,
                candidate_id: NodeId(3),
                last_log_index: 0,
                last_log_term: 0,
            }),
        });

        assert_eq!(
            node.current_term(),
            5,
            "outside a joint transition, recent leader contact must not \
             block an ordinary higher-term RequestVote"
        );
        assert_eq!(node.voted_for(), Some(NodeId(3)));
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 5,
                    voted_for: Some(NodeId(3)),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 5,
                        granted: true,
                    }),
                }
            ]
        );
    }

    #[test]
    fn granting_a_vote_does_not_arm_the_disruption_guard_for_a_later_legitimate_request_vote() {
        // The bug this fix specifically corrects: election_elapsed is reset
        // by granting a vote, which is exactly why the guard cannot be
        // keyed on election_elapsed. leader_contact_elapsed must be
        // unaffected by granting a vote, so a follower that just granted
        // one to candidate A can still correctly evaluate candidate B's
        // legitimate higher-term RequestVote on its own merits right
        // afterward, inside a joint transition where the guard is active.
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(4), NodeId(5)],
            old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
        };
        let log = vec![LogEntry {
            term: 0,
            payload: LogPayload::Configuration(joint_config),
        }];
        let mut node = Node::recover(
            NodeId(2),
            vec![NodeId(1), NodeId(2), NodeId(3)],
            5,
            1,
            HardState::default(),
            log,
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");
        // Never having heard from a leader, leader_contact_elapsed starts
        // at heartbeat_interval (1) -- already stale, so the guard is
        // inert from the very first RequestVote in this test.
        assert_eq!(node.leader_contact_elapsed(), 1);

        // Grant a vote to candidate A in term 5.
        node.step(Event::Step {
            from: NodeId(1),
            message: Message::RequestVote(RequestVote {
                term: 5,
                candidate_id: NodeId(1),
                last_log_index: 1,
                last_log_term: 0,
            }),
        });
        assert_eq!(node.current_term(), 5);
        assert_eq!(node.voted_for(), Some(NodeId(1)));
        // election_elapsed was reset by the grant; leader_contact_elapsed
        // was not (this is the entire point of tracking them separately).
        assert_eq!(node.leader_contact_elapsed(), 1);

        // Candidate B's legitimate higher-term RequestVote must be
        // evaluated normally -- not blocked as though a leader had just
        // made contact, since granting a vote never touched
        // leader_contact_elapsed.
        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::RequestVote(RequestVote {
                term: 6,
                candidate_id: NodeId(3),
                last_log_index: 1,
                last_log_term: 0,
            }),
        });

        assert_eq!(node.current_term(), 6);
        assert_eq!(node.voted_for(), Some(NodeId(3)));
        assert_eq!(
            effects,
            vec![
                Effect::PersistHardState {
                    term: 6,
                    voted_for: Some(NodeId(3)),
                },
                Effect::Send {
                    to: NodeId(3),
                    message: Message::RequestVoteResponse(RequestVoteResponse {
                        term: 6,
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
                payload: LogPayload::Command(Vec::new()),
            },
            LogEntry {
                term: 1,
                payload: LogPayload::Command(Vec::new()),
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
                        payload: LogPayload::Command(vec![1]),
                    },
                    LogEntry {
                        term: 1,
                        payload: LogPayload::Command(vec![2]),
                    },
                ],
                leader_commit: 0,
            }),
        });
        assert_eq!(node.last_log_index(), 2);

        let replacement = LogEntry {
            term: 2,
            payload: LogPayload::Command(vec![9]),
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
                    payload: LogPayload::Command(Vec::new()),
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
                payload: LogPayload::Command(vec![7]),
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
                        payload: LogPayload::Command(vec![1]),
                    },
                    LogEntry {
                        term: 1,
                        payload: LogPayload::Command(vec![2]),
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
                    payload: LogPayload::Command(vec![3]),
                }],
                leader_commit: 3,
            }),
        });

        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![3])
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
                        payload: LogPayload::Command(vec![3])
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
                payload: LogPayload::Command(vec![3])
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
                payload: LogPayload::Command(vec![3])
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
                config: ClusterConfig {
                    voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                    old_voters: None,
                },
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
                config: ClusterConfig {
                    voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                    old_voters: None,
                },
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
                    payload: LogPayload::Command(vec![1]),
                }],
                leader_commit: 0,
            }),
        });
        assert_eq!(node.log().len(), 1);

        // The transferred config differs from the receiver's own prior
        // config (established at construction from `peers`) -- this is the
        // scenario a learner or a very-far-behind node hits in practice: it
        // never saw the Configuration log entries directly, only the
        // compacted-away result, so this InstallSnapshot is its only way to
        // learn current membership.
        let installed_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(2), NodeId(4)],
            old_voters: None,
        };

        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::InstallSnapshot(InstallSnapshot {
                term: 1,
                leader_id: NodeId(2),
                last_included_index: 5,
                last_included_term: 1,
                data: vec![9, 9],
                config: installed_config.clone(),
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
        // The receiver's current_config() now matches the transferred
        // snapshot's config, not the stale one it started with.
        assert_eq!(node.current_config(), &installed_config);
        assert_eq!(
            effects,
            vec![
                Effect::PersistSnapshot {
                    last_included_index: 5,
                    last_included_term: 1,
                    data: vec![9, 9],
                    config: installed_config,
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
                config: ClusterConfig {
                    voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                    old_voters: None,
                },
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
                    config: ClusterConfig {
                        voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                        old_voters: None,
                    },
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
                        payload: LogPayload::Command(vec![1]),
                    },
                    LogEntry {
                        term: 1,
                        payload: LogPayload::Command(vec![2]),
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
                // Deliberately different from the receiver's own config --
                // a no-op must leave snapshot_config/current_config
                // untouched exactly as it already leaves snapshot/
                // snapshot_data/log untouched.
                config: ClusterConfig {
                    voters: vec![NodeId(1), NodeId(2), NodeId(9)],
                    old_voters: None,
                },
            }),
        });

        assert_eq!(node.log().len(), 2);
        assert_eq!(node.snapshot(), SnapshotMetadata::default());
        assert!(node.snapshot_data().is_empty());
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                old_voters: None,
            }
        );
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
                payload: LogPayload::Command(Vec::new()),
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
                payload: LogPayload::Command(Vec::new()),
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
                    payload: LogPayload::Command(vec![9]),
                }],
            }]
        );
        assert_eq!(
            node.log(),
            &[LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![9]),
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
    fn propose_config_change_on_a_single_node_cluster_completes_the_full_joint_to_plain_lifecycle()
    {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Leader);
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1)],
                old_voters: None,
            }
        );

        let (index, effects) = node
            .propose_config_change(vec![NodeId(1)])
            .expect("leader accepts a config change");

        // A single-node cluster reaches both halves of the dual-majority
        // rule trivially, so the same call that proposes the joint entry
        // also drives the automatic C_new follow-up all the way to commit
        // -- the full lifecycle end-to-end, not split across two calls.
        assert_eq!(
            index, 1,
            "the joint entry is index 1 -- there is no bootstrap log entry"
        );
        assert_eq!(
            effects,
            vec![
                Effect::PersistLog {
                    from_index: 1,
                    entries: vec![LogEntry {
                        term: 1,
                        payload: LogPayload::Configuration(ClusterConfig {
                            voters: vec![NodeId(1)],
                            old_voters: Some(vec![NodeId(1)]),
                        }),
                    }],
                },
                Effect::PersistLog {
                    from_index: 2,
                    entries: vec![LogEntry {
                        term: 1,
                        payload: LogPayload::Configuration(ClusterConfig {
                            voters: vec![NodeId(1)],
                            old_voters: None,
                        }),
                    }],
                },
            ]
        );
        assert_eq!(node.commit_index(), 2);
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1)],
                old_voters: None,
            },
            "the automatic C_new follow-up committed too, clearing old_voters again"
        );
        assert_eq!(
            node.role(),
            Role::Leader,
            "self is still a voter under C_new, so no step-down fires"
        );
    }

    #[test]
    fn promoting_a_learner_via_config_change_removes_it_from_learners_and_persists_the_change() {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Leader);

        let add_effects = node.add_learner(NodeId(2));
        assert!(
            add_effects
                .iter()
                .any(|effect| matches!(effect, Effect::PersistLearners { learners } if learners == &[NodeId(2)]))
        );
        assert_eq!(node.learners(), &BTreeSet::from([NodeId(2)]));

        let (_, effects) = node
            .propose_config_change(vec![NodeId(1), NodeId(2)])
            .expect("leader accepts a config change promoting its learner");

        assert!(
            node.learners().is_empty(),
            "a promoted learner must not still be tracked as one"
        );
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::PersistLearners { learners } if learners.is_empty())),
            "the now-empty learner set must be persisted, not just changed in memory"
        );
    }

    #[test]
    fn a_leader_excluded_from_c_new_steps_down_only_once_c_new_commits_not_c_old_new() {
        let mut node = established_leader();
        assert_eq!(node.current_term(), 1);
        assert_eq!(node.voted_for(), Some(NodeId(1)));

        node.propose_config_change(vec![NodeId(2), NodeId(3)])
            .expect("leader accepts a config change that excludes itself");
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(2), NodeId(3)],
                old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
            },
            "the joint config takes effect immediately, even before it commits"
        );

        // The new set {2, 3} needs both members to ack before its own
        // majority is reached -- node 2 alone is not enough yet.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 1,
            }),
        });
        assert_eq!(
            node.commit_index(),
            0,
            "the new set alone (only node 2 of the new voters) has not reached its own majority"
        );
        assert_eq!(node.role(), Role::Leader);

        // Node 3 acking the joint entry now completes both sets' own
        // majorities: it commits, and the automatic C_new follow-up
        // (voters: {2, 3}, old_voters: None) is appended in this same
        // call -- live immediately, even though it hasn't committed yet.
        node.step(Event::Step {
            from: NodeId(3),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 1,
            }),
        });
        assert_eq!(node.commit_index(), 1, "the joint entry has now committed");
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(2), NodeId(3)],
                old_voters: None,
            },
            "C_new takes effect immediately, before it has itself committed"
        );
        assert_eq!(
            node.role(),
            Role::Leader,
            "excluding self from C_new only matters once C_new itself commits"
        );

        // Node 2 acking the follow-up (index 2) alone is not enough -- the
        // new set {2, 3} still needs both.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 2,
            }),
        });
        assert_eq!(node.commit_index(), 1);
        assert_eq!(node.role(), Role::Leader);

        // Node 3 acking the follow-up too finally commits C_new -- only
        // now does the leader actually step down.
        let effects = node.step(Event::Step {
            from: NodeId(3),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: 1,
                success: true,
                match_index: 2,
            }),
        });

        assert_eq!(node.commit_index(), 2);
        assert_eq!(
            node.role(),
            Role::Follower,
            "C_new commits and excludes self -- step down"
        );
        assert_eq!(
            node.current_term(),
            1,
            "this step-down is config-triggered, not term-driven -- the term is unchanged"
        );
        assert_eq!(
            node.voted_for(),
            Some(NodeId(1)),
            "retire_from_leadership must not touch voted_for, unlike step_down"
        );
        // The trailing replicate_to(from) both response handlers otherwise
        // emit unconditionally must be suppressed once a step-down fires
        // inside advance_commit_index in this same batch: a node that just
        // demoted itself must not still emit leader replication traffic.
        assert!(
            effects.is_empty(),
            "no leader-replication Send may appear in the batch that produced the step-down"
        );
    }

    #[test]
    fn proposing_a_config_change_that_adds_a_never_seeded_voter_does_not_panic_or_commit_early() {
        let mut node = established_leader();

        // NodeId(4) was never a member -- become_leader never seeded it
        // into next_index/match_index. advance_commit_index's majority
        // arithmetic must treat it as "not caught up" (index 0) rather
        // than panicking on a missing match_index entry. Deliberately no
        // Event::Tick anywhere in this test: a heartbeat's
        // broadcast_replication sweep -- not advance_commit_index -- is
        // the actually-unsafe path for a never-seeded voter (see
        // propose_config_change's doc comment); asserting past that would
        // require the become_leader/broadcast_replication seeding fix that
        // is explicitly deferred, staged work, not this stage's job.
        let (index, _effects) = node
            .propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)])
            .expect("leader accepts a config change that adds a new voter");

        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
                old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
            }
        );
        // Neither the new set (self alone, of a 4-voter quorum of 3) nor
        // the old set (self alone, of a 3-voter quorum of 2) has reached
        // its own majority yet -- the joint entry must not have committed.
        assert_eq!(node.commit_index(), 0);
        assert_eq!(
            node.role(),
            Role::Leader,
            "no panic, no premature step-down"
        );
        assert_eq!(index, 1);
    }

    #[test]
    fn add_learner_is_a_no_op_when_not_leader() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 5, 2).expect("valid node");

        let effects = node.add_learner(NodeId(4));

        assert!(effects.is_empty());
        assert!(node.learners().is_empty());
    }

    #[test]
    fn add_learner_is_a_no_op_for_an_existing_voter() {
        let mut node = established_leader();

        let effects = node.add_learner(NodeId(2));

        assert!(
            effects.is_empty(),
            "a voter doesn't need learner tracking too -- the two states must not coexist"
        );
        assert!(node.learners().is_empty());
    }

    #[test]
    fn remove_learner_is_a_no_op_when_not_tracked() {
        let mut node = established_leader();

        let effects = node.remove_learner(NodeId(4));

        assert!(effects.is_empty());
    }

    #[test]
    fn remove_learner_untracks_and_persists_the_updated_set() {
        let mut node = established_leader();
        node.add_learner(NodeId(4));

        let effects = node.remove_learner(NodeId(4));

        assert_eq!(
            effects,
            vec![Effect::PersistLearners {
                learners: Vec::new()
            }]
        );
        assert!(node.learners().is_empty());
    }

    #[test]
    fn a_learner_is_replicated_to_but_never_counted_toward_quorum() {
        let mut node = established_leader();

        let effects = node.add_learner(NodeId(4));
        assert_eq!(
            effects,
            vec![Effect::PersistLearners {
                learners: vec![NodeId(4)],
            }]
        );
        assert_eq!(node.learners(), &BTreeSet::from([NodeId(4)]));

        // A heartbeat now reaches the learner too, not just voters --
        // become_leader already ran before the learner existed, so this
        // relies on add_learner's own next_index/match_index seeding.
        let effects = node.step(Event::Tick { next_timeout: 5 });
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Send { to: NodeId(4), .. })),
            "a tracked learner must receive ordinary replication"
        );

        // Even a fabricated, wildly-ahead match_index from the learner
        // must not move commit_index -- it isn't a voter under
        // current_config, so it's outside advance_commit_index's quorum
        // arithmetic entirely, not merely outvoted.
        node.step(Event::Step {
            from: NodeId(4),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: node.current_term(),
                success: true,
                match_index: 1000,
            }),
        });
        assert_eq!(node.commit_index(), 0);
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                old_voters: None,
            },
            "the learner's response must not have touched membership either"
        );
    }

    /// The gap `propose_config_change`'s own doc comment (and 8ffe466's
    /// commit message) documented as known and unreachable-in-practice: a
    /// genuinely new voter added straight into `propose_config_change`
    /// (never having gone through `add_learner` first) was never seeded
    /// into `next_index`/`match_index`, so the next heartbeat sweep would
    /// panic in `replicate_to`'s `.expect(...)`. Adding it as a learner
    /// first -- the on-ramp this stage adds -- closes that gap: this test
    /// deliberately does what the prior stage's own gate test
    /// (`proposing_a_config_change_that_adds_a_never_seeded_voter_does_not_panic_or_commit_early`)
    /// avoided on purpose, driving a real `Event::Tick` afterward.
    #[test]
    fn add_learner_then_promote_via_config_change_then_tick_replicates_without_panicking() {
        let mut node = established_leader();

        node.add_learner(NodeId(4));

        node.propose_config_change(vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)])
            .expect("leader accepts a config change promoting an existing learner");
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)],
                old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
            }
        );

        // No panic here is the actual assertion -- become_leader ran
        // before NodeId(4) was ever a member of anything, so this only
        // survives because add_learner seeded it up front.
        let effects = node.step(Event::Tick { next_timeout: 5 });

        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Send { to: NodeId(4), .. })),
            "the newly promoted member must actually be replicated to, not just committed to config"
        );
    }

    #[test]
    fn a_node_excluded_from_current_config_never_calls_its_own_election() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut node = Node::new(NodeId(1), peers, 3, 1).expect("valid node");

        // A leader's AppendEntries carrying a config that excludes this
        // node takes effect immediately, even before it commits -- the
        // same live-on-append rule any Configuration entry gets. This is
        // the closest reachable stand-in for a genuine not-yet-promoted
        // learner: from its own perspective, a learner is exactly a node
        // whose current_config doesn't list it as a voter.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 1,
                leader_id: NodeId(2),
                prev_log_index: 0,
                prev_log_term: 0,
                entries: vec![LogEntry {
                    term: 1,
                    payload: LogPayload::Configuration(ClusterConfig {
                        voters: vec![NodeId(2), NodeId(3)],
                        old_voters: None,
                    }),
                }],
                leader_commit: 0,
            }),
        });
        assert!(!node.current_config().is_voter(NodeId(1)));

        // Drive well past the election timeout, several times over.
        for _ in 0..20 {
            node.step(Event::Tick { next_timeout: 3 });
        }

        assert_eq!(
            node.role(),
            Role::Follower,
            "an excluded node must never self-elect"
        );
        assert_eq!(
            node.current_term(),
            1,
            "no election attempt was ever made, so the term never bumped past what the AppendEntries carried"
        );
    }

    #[test]
    fn a_leader_that_just_retired_never_re_campaigns() {
        let mut node = established_leader();
        node.propose_config_change(vec![NodeId(2), NodeId(3)])
            .expect("leader accepts a config change that excludes itself");

        // Drive the joint config, then the automatic C_new follow-up, all
        // the way to commit -- both new-set members acking twice, mirroring
        // a_leader_excluded_from_c_new_steps_down_only_once_c_new_commits_not_c_old_new.
        for match_index in [1, 2] {
            node.step(Event::Step {
                from: NodeId(2),
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: node.current_term(),
                    success: true,
                    match_index,
                }),
            });
            node.step(Event::Step {
                from: NodeId(3),
                message: Message::AppendEntriesResponse(AppendEntriesResponse {
                    term: node.current_term(),
                    success: true,
                    match_index,
                }),
            });
        }
        assert_eq!(node.role(), Role::Follower, "retired once C_new committed");
        let term_after_retirement = node.current_term();

        for _ in 0..20 {
            node.step(Event::Tick { next_timeout: 3 });
        }

        assert_eq!(
            node.role(),
            Role::Follower,
            "a retired leader must never re-campaign on its own"
        );
        assert_eq!(
            node.current_term(),
            term_after_retirement,
            "no election attempt means no term bump"
        );
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
                    config: ClusterConfig {
                        voters: vec![NodeId(1)],
                        old_voters: None,
                    },
                },
                Effect::PersistLog {
                    from_index: 3,
                    entries: vec![LogEntry {
                        term: 1,
                        payload: LogPayload::Command(vec![3]),
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
                payload: LogPayload::Command(vec![3])
            }]
        );
        assert_eq!(node.last_log_index(), 3);
        assert_eq!(node.entry_at(2), None);
        assert_eq!(
            node.entry_at(3),
            Some(&LogEntry {
                term: 1,
                payload: LogPayload::Command(vec![3])
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
                    config: ClusterConfig {
                        voters: vec![NodeId(1)],
                        old_voters: None,
                    },
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
    fn compact_leaves_snapshot_config_untouched_when_the_active_config_entry_is_above_the_boundary()
    {
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.propose(vec![1]);
        assert_eq!(node.commit_index(), 1);

        // Appends a joint Configuration entry adding NodeId(2). It takes
        // effect immediately (current_config() is live, not commit-gated),
        // but stays uncommitted here: dual-majority requires NodeId(2)'s
        // own ack for the new-side quorum, which never arrives in this
        // test -- commit_index stays at 1.
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(2)],
            old_voters: Some(vec![NodeId(1)]),
        };
        node.propose_config_change(vec![NodeId(1), NodeId(2)])
            .expect("leader accepts propose_config_change");
        assert_eq!(node.current_config(), &joint_config);
        assert_eq!(node.commit_index(), 1);

        // Compacting only through the still-committed index 1 -- the joint
        // config entry at index 2 is above this boundary and untouched by
        // the drain, so the config *at the boundary being created* is the
        // node's original bootstrap config, not the not-yet-committed
        // joint one: snapshot_config must reflect that, not whatever
        // current_config happens to be right now. If index 2 is later
        // truncated away by a conflicting AppendEntries (it never
        // committed, so it's legitimately discardable), snapshot_config is
        // exactly what current_config falls back to -- it must not hold a
        // config from an entry that turned out not to survive.
        let bootstrap_config = ClusterConfig {
            voters: vec![NodeId(1)],
            old_voters: None,
        };
        let effects = node.compact(1, vec![9]).expect("1 is committed");

        assert_eq!(
            effects,
            vec![
                Effect::PersistSnapshot {
                    last_included_index: 1,
                    last_included_term: 1,
                    data: vec![9],
                    config: bootstrap_config,
                },
                Effect::PersistLog {
                    from_index: 2,
                    entries: vec![LogEntry {
                        term: 1,
                        payload: LogPayload::Configuration(joint_config.clone()),
                    }],
                },
            ]
        );
        // The config entry itself survived the drain (it's above the new
        // boundary), so current_config() is unaffected by compacting.
        assert_eq!(node.current_config(), &joint_config);
    }

    #[test]
    fn a_config_entry_truncated_after_compaction_falls_back_to_the_boundarys_real_config_not_the_discarded_one()
     {
        // The scenario the fix above exists for: an uncommitted
        // Configuration entry above the compaction boundary is later
        // truncated away by a conflicting AppendEntries (a new leader's
        // log didn't include it). current_config() must fall back to
        // whatever config was genuinely active at the snapshot boundary --
        // here, the original bootstrap config -- not the discarded entry's
        // config, even though that entry was current_config() at the
        // moment compact() ran.
        let mut node = Node::new(NodeId(1), vec![NodeId(1)], 1, 1).expect("valid node");
        node.step(Event::Tick { next_timeout: 5 });
        node.propose(vec![1]);
        assert_eq!(node.commit_index(), 1);

        node.propose_config_change(vec![NodeId(1), NodeId(2)])
            .expect("leader accepts propose_config_change");
        assert_eq!(node.commit_index(), 1);

        node.compact(1, vec![9]).expect("1 is committed");
        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1), NodeId(2)],
                old_voters: Some(vec![NodeId(1)]),
            },
            "the uncommitted config entry survives the drain, so it's still current"
        );

        // A conflicting AppendEntries from a legitimate higher-term leader
        // (NodeId(2), already a voter under the joint config so it clears
        // step()'s is_member check) truncates away the uncommitted config
        // entry at index 2, replacing it with an ordinary command entry
        // the majority actually agreed on instead.
        node.step(Event::Step {
            from: NodeId(2),
            message: Message::AppendEntries(AppendEntries {
                term: 2,
                leader_id: NodeId(2),
                prev_log_index: 1,
                prev_log_term: 1,
                entries: vec![LogEntry {
                    term: 2,
                    payload: LogPayload::Command(vec![2]),
                }],
                leader_commit: 1,
            }),
        });

        assert_eq!(
            node.current_config(),
            &ClusterConfig {
                voters: vec![NodeId(1)],
                old_voters: None,
            },
            "falling back to snapshot_config after truncation must land on the \
             boundary's real config, not the discarded entry's"
        );
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
                        payload: LogPayload::Command(vec![2]),
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
                            payload: LogPayload::Command(vec![1]),
                        },
                        LogEntry {
                            term: 1,
                            payload: LogPayload::Command(vec![2]),
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
                        config: ClusterConfig {
                            voters: vec![NodeId(1), NodeId(2), NodeId(3)],
                            old_voters: None,
                        },
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

    #[test]
    fn a_joint_election_requires_a_majority_of_both_the_old_and_new_voters() {
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(4), NodeId(5)],
            old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
        };
        let log = vec![LogEntry {
            term: 0,
            payload: LogPayload::Configuration(joint_config.clone()),
        }];
        let mut node = Node::recover(
            NodeId(1),
            vec![NodeId(1)],
            3,
            3,
            HardState::default(),
            log,
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");
        assert_eq!(node.current_config(), &joint_config);

        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        let effects = node.step(Event::Tick { next_timeout: 5 });
        assert_eq!(node.role(), Role::Candidate);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, Effect::Send { to: NodeId(4), .. }))
        );

        // A majority of the new set alone (self + 4) is not enough: the old
        // set (1, 2, 3) has granted nothing beyond self's own vote.
        node.step(Event::Step {
            from: NodeId(4),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: node.current_term(),
                granted: true,
            }),
        });
        assert_eq!(
            node.role(),
            Role::Candidate,
            "new-set majority alone must not win a joint election"
        );

        // Once the old set also reaches its own majority (self + 2), the
        // election is won.
        let effects = node.step(Event::Step {
            from: NodeId(2),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: node.current_term(),
                granted: true,
            }),
        });

        assert_eq!(node.role(), Role::Leader);
        assert!(!effects.is_empty());
    }

    /// Distinguishes a correct (minimum-of-both-thresholds) dual-majority
    /// commit rule from an incorrect (maximum-of-both-thresholds) one: a
    /// `max`-based implementation would already commit after the *first*
    /// response below (new-set majority alone), which is exactly the
    /// split-brain scenario joint consensus exists to rule out, since the
    /// old set hasn't independently agreed to anything yet.
    #[test]
    fn a_joint_commit_requires_match_progress_on_both_the_old_and_new_voters() {
        let joint_config = ClusterConfig {
            voters: vec![NodeId(1), NodeId(4), NodeId(5)],
            old_voters: Some(vec![NodeId(1), NodeId(4)]),
        };
        let log = vec![LogEntry {
            term: 0,
            payload: LogPayload::Configuration(joint_config.clone()),
        }];
        let mut node = Node::recover(
            NodeId(1),
            vec![NodeId(1)],
            3,
            3,
            HardState::default(),
            log,
            Snapshot::default(),
            Vec::new(),
        )
        .expect("valid node");

        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        node.step(Event::Tick { next_timeout: 5 });
        // A single grant from NodeId(4) satisfies both the new-set quorum
        // (self + 4, of 3) and the old-set quorum (self + 4, of 2 -- the
        // old set here is exactly {1, 4}, so it requires both members).
        node.step(Event::Step {
            from: NodeId(4),
            message: Message::RequestVoteResponse(RequestVoteResponse {
                term: node.current_term(),
                granted: true,
            }),
        });
        assert_eq!(node.role(), Role::Leader);

        let (index, _effects) = node.propose(vec![9]).expect("leader accepts propose");
        assert_eq!(index, 2, "index 1 is the joint config entry itself");
        assert_eq!(node.commit_index(), 0);

        // NodeId(5) is in the new set only. Its catching up alone reaches
        // new-set majority (self + 5, of 3) but the old set (1, 4) still
        // has only self's own progress -- below its own quorum of 2.
        node.step(Event::Step {
            from: NodeId(5),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: node.current_term(),
                success: true,
                match_index: index,
            }),
        });
        assert_eq!(
            node.commit_index(),
            0,
            "new-set majority alone must not commit a joint entry"
        );

        // NodeId(4) catching up now completes the old set's own majority
        // too (self + 4, of 2) -- both thresholds are satisfied, so the
        // entry finally commits.
        node.step(Event::Step {
            from: NodeId(4),
            message: Message::AppendEntriesResponse(AppendEntriesResponse {
                term: node.current_term(),
                success: true,
                match_index: index,
            }),
        });
        assert_eq!(node.commit_index(), index);
    }
}
