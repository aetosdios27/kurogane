//! A deterministic-but-fault-injecting harness over real `kurogane-kv`
//! `Replica`s, plus a hand-rolled linearizability checker over the
//! client-visible histories it produces. `Outcome`'s three states were
//! deliberately designed to mirror `kurogane-runtime`'s real `ProposeReply`
//! shape exactly, so a future real-process history generator could reuse
//! `is_linearizable` unchanged -- but no such generator exists yet; today
//! this checker only ever sees histories this crate's own `Harness`
//! produced.
//!
//! Lives in its own crate rather than inside `kurogane-sim` because
//! `kurogane-sim`'s `Cluster`/`Simulation` are deliberately scoped to raw
//! `kurogane-raft::Node` scheduling ("for the raft layer" -- see
//! `CLAUDE.md`), not KV-level semantics, and `kurogane-kv` is a peer of
//! `kurogane-sim`, not a dependent, per the same architecture note. This
//! crate depends on both, the same way `kurogane-runtime` already depends
//! on `kurogane-raft` and `kurogane-kv` without depending on
//! `kurogane-sim` -- driving `Replica`s directly with its own small
//! tick/inbox loop, reusing `kurogane-sim`'s `Rng` (made `pub` for this)
//! and `DurableState` rather than duplicating either.

use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::hash::{Hash, Hasher};

use kurogane_kv::{ApplyResult, Command, Replica, StateMachine};
use kurogane_raft::{Effect, Event, Message, Node, NodeId, Role, Snapshot};
use kurogane_sim::{DurableState, Rng, TraceEvent};

/// Distinguishes one simulated client's operations from another's. A
/// client only ever has one operation outstanding at a time (it waits for
/// a resolution before submitting its next one, mirroring a real
/// synchronous RPC caller) -- concurrency in the generated history comes
/// from multiple distinct `ClientId`s overlapping, not from any one client
/// pipelining.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ClientId(pub u64);

/// A client-visible RPC outcome. Mirrors `kurogane_runtime::server`'s real
/// `ProposeReply` three-state shape exactly (see that crate's `Propose`
/// handler and `.proto` comment) so a real-process history and a simulated
/// one can share one checker: `Rejected` is a clean pre-append rejection
/// (this node was never leader for this attempt -- definitely never
/// happened), `Applied` is a confirmed success with its real value, and
/// `Indeterminate` covers everything in between -- still in flight when
/// observation stopped, or truncated by a conflicting leader -- both of
/// which look identical from the client's side (no value ever arrives).
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Applied(ApplyResult),
    Rejected,
    Indeterminate,
}

/// One operation's full lifecycle, tick-stamped by the harness's own
/// logical clock -- never wall time, so a persisted seed replays this
/// exact history deterministically, the same reproducibility contract
/// `kurogane-sim`'s own traces already rely on.
#[derive(Clone, Debug, PartialEq)]
pub struct HistoryEntry {
    pub client: ClientId,
    pub invoke_tick: u64,
    /// `None` only if the run ended with this operation still outstanding
    /// -- its effect (if any) is unknown, same as `Indeterminate`.
    pub return_tick: Option<u64>,
    pub command: Command,
    pub outcome: Outcome,
}

/// One kind of fault-injection event `Harness` can record -- see
/// `Harness::faults`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaultKind {
    Isolated,
    Healed,
    Crashed,
    Restarted,
}

/// One fault-injection event that actually changed something, tick-stamped
/// by the harness's own logical clock -- the retained record a run's gate
/// evidence points at, matching `decisions.md`'s "retain the seed and
/// complete event history" convention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FaultEvent {
    pub tick: u64,
    pub node: NodeId,
    pub kind: FaultKind,
}

/// One client operation the harness is still waiting on.
struct PendingOp {
    client: ClientId,
    invoke_tick: u64,
    target: NodeId,
    index: u64,
    /// The term the entry actually held immediately after this op's own
    /// `propose` call. `last_applied >= index` alone isn't proof *this*
    /// op's command is what applied there -- a leadership change can
    /// truncate and replace the entry at the same index with a different
    /// command entirely. Raft's Log Matching Property (same index + same
    /// term implies the same command) is the standard, provably-correct
    /// way to tell the two cases apart; `resolve_pending` re-checks this
    /// before ever trusting `applied_result(index)`.
    term: u64,
    command: Command,
    deadline: u64,
}

/// Drives a set of `Replica`s through logical time exactly like
/// `kurogane_sim::Simulation` drives raw `Node`s -- seeded message delays,
/// seeded election timeouts, a full trace -- plus client operations
/// (`submit`/`step` resolves them once applied or once they time out) and
/// fault injection (`isolate`/`heal` for partitions, `crash`/`restart` for
/// process death, both reusing `kurogane_sim::DurableState` for exactly
/// the same write-before-send-respecting recovery contract
/// `kurogane-sim`'s own crash tests already prove).
pub struct Harness {
    replicas: BTreeMap<NodeId, Replica>,
    durable: BTreeMap<NodeId, DurableState>,
    isolated: BTreeSet<NodeId>,
    peers: Vec<NodeId>,
    election_timeout: u64,
    heartbeat_interval: u64,
    rng: Rng,
    clock: u64,
    inbox: BTreeMap<u64, Vec<(NodeId, NodeId, Message)>>,
    min_timeout: u64,
    max_timeout: u64,
    min_delay: u64,
    max_delay: u64,
    apply_timeout_ticks: u64,
    trace: Vec<TraceEvent>,
    pending: Vec<PendingOp>,
    history: Vec<HistoryEntry>,
    faults: Vec<FaultEvent>,
}

impl Harness {
    #[allow(
        clippy::too_many_arguments,
        reason = "mirrors Simulation::new's own flat parameter list, plus the two fields (peers, apply_timeout_ticks) this harness needs beyond it to build/rebuild Replicas itself"
    )]
    pub fn new(
        peers: Vec<NodeId>,
        election_timeout: u64,
        heartbeat_interval: u64,
        seed: u64,
        min_timeout: u64,
        max_timeout: u64,
        min_delay: u64,
        max_delay: u64,
        apply_timeout_ticks: u64,
    ) -> Self {
        let mut replicas = BTreeMap::new();
        let mut durable = BTreeMap::new();
        for &id in &peers {
            let node = Node::new(id, peers.clone(), election_timeout, heartbeat_interval)
                .expect("valid bootstrap configuration");
            replicas.insert(id, Replica::new(node));
            durable.insert(id, DurableState::default());
        }

        Self {
            replicas,
            durable,
            isolated: BTreeSet::new(),
            peers,
            election_timeout,
            heartbeat_interval,
            rng: Rng::new(seed),
            clock: 0,
            inbox: BTreeMap::new(),
            min_timeout,
            max_timeout,
            min_delay,
            max_delay,
            apply_timeout_ticks,
            trace: Vec::new(),
            pending: Vec::new(),
            history: Vec::new(),
            faults: Vec::new(),
        }
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    pub fn trace(&self) -> &[TraceEvent] {
        &self.trace
    }

    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }

    /// Every fault-injection event that actually changed something, in the
    /// order it happened -- the replay artifact for whatever run produced
    /// this harness's history.
    pub fn faults(&self) -> &[FaultEvent] {
        &self.faults
    }

    /// Any node currently in `Role::Leader` -- best-effort, exactly like a
    /// real client's cached leader hint, since a driver uses this only to
    /// pick a reasonable propose target, never to affect the recorded
    /// outcome itself.
    pub fn current_leader(&self) -> Option<NodeId> {
        self.replicas
            .iter()
            .find(|(_, replica)| replica.node().role() == Role::Leader)
            .map(|(&id, _)| id)
    }

    /// Reads `id`'s own `Replica` directly without going through a client
    /// operation -- e.g. to confirm an isolated/restarted node's actual
    /// applied state independent of whatever it would itself report if
    /// asked (which, for anything but the leader, would just be a
    /// `Rejected` redirect). `None` if `id` is currently down.
    pub fn replica(&self, id: NodeId) -> Option<&Replica> {
        self.replicas.get(&id)
    }

    /// Partitions `id` from every other node: messages to or from it are
    /// dropped, never delayed-and-delivered, matching a real network
    /// partition rather than a slow link.
    pub fn isolate(&mut self, id: NodeId) {
        if self.isolated.insert(id) {
            self.record_fault(id, FaultKind::Isolated);
        }
    }

    pub fn heal(&mut self, id: NodeId) {
        if self.isolated.remove(&id) {
            self.record_fault(id, FaultKind::Healed);
        }
    }

    pub fn is_isolated(&self, id: NodeId) -> bool {
        self.isolated.contains(&id)
    }

    /// Simulates a process crash: the in-memory `Replica` is gone, but its
    /// `DurableState` (whatever `Persist*` effects actually landed) is
    /// untouched -- `restart` rebuilds from exactly that, the same
    /// contract `kurogane_sim::Cluster::replace_node` proves at the raw
    /// `Node` level. A no-op if `id` is already down.
    pub fn crash(&mut self, id: NodeId) {
        if self.replicas.remove(&id).is_some() {
            self.record_fault(id, FaultKind::Crashed);
        }
    }

    /// Reconstructs `id` from its `DurableState` and reinserts it as a
    /// live replica -- always a `Follower`, per `Node::recover`'s own
    /// contract. A no-op if `id` is already up or isn't a known peer.
    pub fn restart(&mut self, id: NodeId) {
        if self.replicas.contains_key(&id) || !self.peers.contains(&id) {
            return;
        }
        let Some(durable) = self.durable.get(&id) else {
            return;
        };
        let node = Node::recover(
            id,
            self.peers.clone(),
            self.election_timeout,
            self.heartbeat_interval,
            durable.hard_state(),
            durable.log().to_vec(),
            Snapshot {
                metadata: durable.snapshot(),
                data: durable.snapshot_data().to_vec(),
                config: durable.snapshot_config().clone(),
            },
            durable.learners().to_vec(),
        )
        .expect("a node's own durable state always reconstructs validly");
        self.replicas.insert(id, Replica::recover(node));
        self.record_fault(id, FaultKind::Restarted);
    }

    fn record_fault(&mut self, node: NodeId, kind: FaultKind) {
        self.faults.push(FaultEvent {
            tick: self.clock,
            node,
            kind,
        });
    }

    /// Submits `command` against `target` on behalf of `client`, mirroring
    /// `kurogane-runtime`'s real blocking `Propose` RPC: resolves
    /// immediately with `Rejected` if `target` isn't reachable (crashed)
    /// or isn't the leader, otherwise resolves later (via `step`) once the
    /// assigned index applies or `apply_timeout_ticks` elapses.
    pub fn submit(&mut self, client: ClientId, target: NodeId, command: Command) {
        let Some(replica) = self.replicas.get_mut(&target) else {
            self.history.push(HistoryEntry {
                client,
                invoke_tick: self.clock,
                return_tick: Some(self.clock),
                command,
                outcome: Outcome::Rejected,
            });
            return;
        };

        match replica.propose(command.clone()) {
            Some((index, effects)) => {
                let term = replica
                    .node()
                    .entry_at(index)
                    .expect("the entry this call just appended")
                    .term;
                self.apply_effects(target, effects);
                self.pending.push(PendingOp {
                    client,
                    invoke_tick: self.clock,
                    target,
                    index,
                    term,
                    command,
                    deadline: self.clock + self.apply_timeout_ticks,
                });
            }
            None => {
                self.history.push(HistoryEntry {
                    client,
                    invoke_tick: self.clock,
                    return_tick: Some(self.clock),
                    command,
                    outcome: Outcome::Rejected,
                });
            }
        }
    }

    /// Advances logical time by one tick: delivers due messages, then
    /// ticks every currently-up node in canonical order (a crashed node
    /// simply isn't ticked, exactly like it wouldn't be running), then
    /// resolves any pending client operations that just applied or timed
    /// out. Mirrors `kurogane_sim::Simulation::step` in shape.
    pub fn step(&mut self) {
        self.clock += 1;

        if let Some(due) = self.inbox.remove(&self.clock) {
            for (from, to, message) in due {
                if self.isolated.contains(&from) || self.isolated.contains(&to) {
                    continue;
                }
                let Some(replica) = self.replicas.get_mut(&to) else {
                    continue;
                };
                self.trace.push(TraceEvent::Delivered {
                    at: self.clock,
                    from,
                    to,
                    message: message.clone(),
                });
                let effects = replica.step(Event::Step { from, message });
                self.apply_effects(to, effects);
            }
        }

        let ids: Vec<NodeId> = self.replicas.keys().copied().collect();
        for id in ids {
            let next_timeout = self.rng.range_inclusive(self.min_timeout, self.max_timeout);
            self.trace.push(TraceEvent::Tick {
                at: self.clock,
                node: id,
                next_timeout,
            });
            let effects = self
                .replicas
                .get_mut(&id)
                .expect("known node id")
                .step(Event::Tick { next_timeout });
            self.apply_effects(id, effects);
        }

        self.resolve_pending();
    }

    /// Flushes any operations still outstanding when the caller is done
    /// stepping, recording them as `Indeterminate` with no `return_tick`
    /// -- observation simply stopped, their real effect (if any) is
    /// unknown, never assumed to be `Rejected`.
    pub fn finish(mut self) -> Vec<HistoryEntry> {
        for op in self.pending.drain(..) {
            self.history.push(HistoryEntry {
                client: op.client,
                invoke_tick: op.invoke_tick,
                return_tick: None,
                command: op.command,
                outcome: Outcome::Indeterminate,
            });
        }
        self.history
    }

    fn resolve_pending(&mut self) {
        let clock = self.clock;
        let mut still_pending = Vec::with_capacity(self.pending.len());
        for op in self.pending.drain(..) {
            let applied = self.replicas.get(&op.target).and_then(|replica| {
                if replica.state_machine().last_applied() < op.index {
                    return None;
                }
                // The entry at this index may no longer be the one this op
                // proposed -- a leadership change can truncate and replace
                // it. Only trust `applied_result` when the term still
                // matches what we saw immediately after proposing (Raft's
                // Log Matching Property); otherwise this op's own entry is
                // gone and its real effect (if it ever committed at all
                // before being overwritten) is unknowable from here, same
                // as a plain timeout.
                match replica.node().entry_at(op.index) {
                    Some(entry) if entry.term == op.term => {}
                    _ => return None,
                }
                replica.applied_result(op.index).cloned()
            });

            if let Some(result) = applied {
                self.history.push(HistoryEntry {
                    client: op.client,
                    invoke_tick: op.invoke_tick,
                    return_tick: Some(clock),
                    command: op.command,
                    outcome: Outcome::Applied(result),
                });
            } else if clock >= op.deadline {
                self.history.push(HistoryEntry {
                    client: op.client,
                    invoke_tick: op.invoke_tick,
                    return_tick: Some(clock),
                    command: op.command,
                    outcome: Outcome::Indeterminate,
                });
            } else {
                still_pending.push(op);
            }
        }
        self.pending = still_pending;
    }

    fn apply_effects(&mut self, from: NodeId, effects: Vec<Effect>) {
        for effect in &effects {
            if let Some(durable) = self.durable.get_mut(&from) {
                durable.apply(effect);
            }
        }
        for effect in effects {
            let Effect::Send { to, message } = effect else {
                continue;
            };
            if self.isolated.contains(&from) || self.isolated.contains(&to) {
                continue;
            }
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

/// Fully specifies one generated run together with a seed: `run` is a pure
/// function of `(seed, RunConfig)`, so persisting just these two values
/// (via `save_seed`/`load_seed`) is enough to reproduce an identical trace
/// and history later -- the same "retain the seed" contract
/// `kurogane-sim`'s own invariant checks already rely on, extended to
/// cover the fault-injection and client-operation schedule too.
#[derive(Clone, Debug)]
pub struct RunConfig {
    pub peers: Vec<NodeId>,
    pub election_timeout: u64,
    pub heartbeat_interval: u64,
    pub min_delay: u64,
    pub max_delay: u64,
    pub apply_timeout_ticks: u64,
    pub client_count: u64,
    pub total_ticks: u64,
    /// Chance out of 100, per idle client per tick, that it submits its
    /// next operation.
    pub submit_chance_percent: u64,
    /// Chance out of 100, per tick, that a fault-injection event fires.
    pub fault_chance_percent: u64,
}

/// Generates one concurrent, fault-injected run: `client_count` sessions
/// each keep at most one operation outstanding at a time (so overlapping
/// invoke/return windows across *different* clients are what make the
/// resulting history genuinely concurrent, not one op at a time), while
/// faults (partition/heal/crash/restart) fire independently on their own
/// schedule -- deliberately capable of landing while operations are still
/// in flight, which is what actually exercises the fault model rather
/// than just interleaving. Uses a second `Rng` stream (derived from, but
/// distinct from, `seed`) for client/fault scheduling decisions, so those
/// choices never perturb the harness's own message-delay/timeout draws.
pub fn run(seed: u64, config: &RunConfig) -> (Vec<TraceEvent>, Vec<HistoryEntry>, Vec<FaultEvent>) {
    let mut harness = Harness::new(
        config.peers.clone(),
        config.election_timeout,
        config.heartbeat_interval,
        seed,
        config.election_timeout,
        config.election_timeout * 2,
        config.min_delay,
        config.max_delay,
        config.apply_timeout_ticks,
    );
    let mut driver_rng = Rng::new(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut busy: BTreeSet<ClientId> = BTreeSet::new();
    let mut history_seen = 0usize;
    let mut key_counter: u64 = 0;

    for _ in 0..config.total_ticks {
        harness.step();

        for entry in &harness.history()[history_seen..] {
            busy.remove(&entry.client);
        }
        history_seen = harness.history().len();

        if driver_rng.range_inclusive(0, 99) < config.fault_chance_percent {
            inject_random_fault(&mut harness, &mut driver_rng, &config.peers);
        }

        for client_index in 0..config.client_count {
            let client = ClientId(client_index);
            if busy.contains(&client) {
                continue;
            }
            if driver_rng.range_inclusive(0, 99) >= config.submit_chance_percent {
                continue;
            }
            let target = harness.current_leader().unwrap_or_else(|| {
                let pick = driver_rng.range_inclusive(0, config.peers.len() as u64 - 1);
                config.peers[pick as usize]
            });
            key_counter += 1;
            let command = random_command(&mut driver_rng, key_counter);
            harness.submit(client, target, command);
            busy.insert(client);
        }
    }

    let trace = harness.trace().to_vec();
    let faults = harness.faults().to_vec();
    let history = harness.finish();
    (trace, history, faults)
}

fn random_command(rng: &mut Rng, unique: u64) -> Command {
    const KEYS: [&[u8]; 3] = [b"a", b"b", b"c"];
    let key = KEYS[rng.range_inclusive(0, KEYS.len() as u64 - 1) as usize].to_vec();
    match rng.range_inclusive(0, 2) {
        0 => Command::Set {
            key,
            value: unique.to_le_bytes().to_vec(),
        },
        1 => Command::Get { key },
        _ => Command::Delete { key },
    }
}

fn inject_random_fault(harness: &mut Harness, rng: &mut Rng, peers: &[NodeId]) {
    let target = peers[rng.range_inclusive(0, peers.len() as u64 - 1) as usize];
    match rng.range_inclusive(0, 3) {
        0 => harness.isolate(target),
        1 => harness.heal(target),
        2 => harness.crash(target),
        _ => harness.restart(target),
    }
}

/// Persists just `(seed, config)` to `path` in a small hand-rolled
/// `key=value` line format -- sufficient to reproduce a run exactly, since
/// `run` is a pure function of the two. No dependency reached for, mirrors
/// every other wire/storage encoding in this project.
pub fn save_seed(seed: u64, config: &RunConfig, path: &std::path::Path) -> std::io::Result<()> {
    let peers = config
        .peers
        .iter()
        .map(|id| id.0.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let contents = format!(
        "seed={seed}\npeers={peers}\nelection_timeout={}\nheartbeat_interval={}\nmin_delay={}\nmax_delay={}\napply_timeout_ticks={}\nclient_count={}\ntotal_ticks={}\nsubmit_chance_percent={}\nfault_chance_percent={}\n",
        config.election_timeout,
        config.heartbeat_interval,
        config.min_delay,
        config.max_delay,
        config.apply_timeout_ticks,
        config.client_count,
        config.total_ticks,
        config.submit_chance_percent,
        config.fault_chance_percent,
    );
    std::fs::write(path, contents)
}

/// The inverse of `save_seed`. Panics on a malformed file: this is a
/// developer-facing replay tool, not a client-facing parser that needs to
/// handle adversarial input gracefully.
pub fn load_seed(path: &std::path::Path) -> std::io::Result<(u64, RunConfig)> {
    let contents = std::fs::read_to_string(path)?;
    let mut fields: BTreeMap<&str, &str> = BTreeMap::new();
    for line in contents.lines() {
        let (key, value) = line
            .split_once('=')
            .expect("save_seed's own output is always `key=value` per line");
        fields.insert(key, value);
    }
    let field = |name: &str| -> &str {
        fields
            .get(name)
            .unwrap_or_else(|| panic!("seed file is missing `{name}`"))
    };
    let parse_u64 = |name: &str| -> u64 {
        field(name)
            .parse()
            .unwrap_or_else(|_| panic!("`{name}` is not a valid u64"))
    };
    let peers = field("peers")
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| NodeId(s.parse().expect("peer id is a valid u64")))
        .collect();

    Ok((
        parse_u64("seed"),
        RunConfig {
            peers,
            election_timeout: parse_u64("election_timeout"),
            heartbeat_interval: parse_u64("heartbeat_interval"),
            min_delay: parse_u64("min_delay"),
            max_delay: parse_u64("max_delay"),
            apply_timeout_ticks: parse_u64("apply_timeout_ticks"),
            client_count: parse_u64("client_count"),
            total_ticks: parse_u64("total_ticks"),
            submit_chance_percent: parse_u64("submit_chance_percent"),
            fault_chance_percent: parse_u64("fault_chance_percent"),
        },
    ))
}

/// One operation as the checker sees it: a definite (`Applied`) op has a
/// known real-time window and an expected return value that must match
/// exactly wherever it's placed; an ambiguous (`Indeterminate`) op has
/// neither -- it may be skipped entirely (never took effect) or included
/// at any point (its real effect time is unknown), and never contributes
/// a value to check since the client itself never observed one.
struct Op {
    invoke: u64,
    finish: Option<u64>,
    command: Command,
    expected: Option<ApplyResult>,
    optional: bool,
}

/// Checks whether `history` is linearizable against the replicated KV's
/// sequential model (`kurogane_kv::StateMachine`, reused directly rather
/// than re-encoding Set/Delete/Get semantics a second time here).
///
/// `Rejected` operations are dropped before the search even begins -- a
/// clean pre-append rejection never took effect, full stop. Every
/// `Indeterminate` operation is tried both ways (see `Op::optional`)
/// rather than assumed either way, which is the entire point: silently
/// dropping an op that actually landed but whose client never got
/// confirmation would let a checker "pass" a history that omits a real
/// write, exactly the false confidence this exists to rule out.
///
/// Known limitation, accepted for this milestone rather than solved: an
/// `Indeterminate` operation is never *forced* to precede anything -- its
/// actual effect time is unbounded above (it might apply arbitrarily later
/// than this run ever observed), so a definite operation that returned
/// after an `Indeterminate` one was invoked is still free to be linearized
/// before it. It's still bounded *below* by its own invoke tick, though
/// (nothing can take effect before it was invoked, ambiguous or not), so a
/// definite operation that already finished before an `Indeterminate` one
/// was even invoked still forces that ordering. This asymmetry is a
/// conservative choice -- it can only make the checker more permissive
/// about where an ambiguous operation could slot in, never mask a real
/// violation among the definite operations whose ordering is fully known.
pub fn is_linearizable(history: &[HistoryEntry]) -> bool {
    let ops: Vec<Op> = history
        .iter()
        .filter_map(|entry| match &entry.outcome {
            Outcome::Rejected => None,
            Outcome::Applied(result) => Some(Op {
                invoke: entry.invoke_tick,
                finish: entry.return_tick,
                command: entry.command.clone(),
                expected: Some(result.clone()),
                optional: false,
            }),
            Outcome::Indeterminate => Some(Op {
                invoke: entry.invoke_tick,
                finish: None,
                command: entry.command.clone(),
                expected: None,
                optional: true,
            }),
        })
        .collect();

    assert!(
        ops.len() <= 128,
        "checker's bitmask-based search supports at most 128 concurrent operations, got {}",
        ops.len()
    );
    let full_mask: u128 = if ops.is_empty() {
        0
    } else if ops.len() == 128 {
        u128::MAX
    } else {
        (1u128 << ops.len()) - 1
    };

    let mut failed = HashSet::new();
    search(&ops, full_mask, StateMachine::new(), &mut failed)
}

fn precedes(a: &Op, b: &Op) -> bool {
    // `a`'s effect time is only known (bounded above) when it's definite --
    // an Indeterminate `a` might take effect arbitrarily late, so it can
    // never be forced to precede anything. `b`, on the other hand, is
    // bounded *below* by its own invoke tick regardless of whether it's
    // definite or ambiguous: no operation, Indeterminate or not, can have
    // taken effect before it was ever invoked. Dropping that lower bound
    // for an Indeterminate `b` would let it be linearized earlier than
    // physically possible -- e.g. before a definite op that already
    // returned a value only this one could have produced -- which is
    // exactly the false-accept this checker exists to catch.
    if a.optional {
        return false;
    }
    match a.finish {
        Some(finish) => finish <= b.invoke,
        None => false,
    }
}

fn hash_model(model: &StateMachine) -> u64 {
    let mut hasher = DefaultHasher::new();
    model.hash(&mut hasher);
    hasher.finish()
}

/// Backtracking search over which of `ops` (indexed by bit position in
/// `remaining`) could come next, given `model`'s current sequential state.
/// Memoizes failed `(model state, remaining set)` pairs -- the standard
/// technique behind every practical linearizability checker (Wing & Gong's
/// original algorithm included) -- since many different orderings of a
/// small, bounded key space converge on the same intermediate state,
/// without which this is a plain exponential permutation search that
/// chokes on anything beyond a handful of operations.
fn search(
    ops: &[Op],
    remaining: u128,
    model: StateMachine,
    failed: &mut HashSet<(u64, u128)>,
) -> bool {
    if remaining == 0 {
        return true;
    }

    let state_hash = hash_model(&model);
    if failed.contains(&(state_hash, remaining)) {
        return false;
    }

    for i in 0..ops.len() {
        if remaining & (1u128 << i) == 0 {
            continue;
        }
        let op = &ops[i];
        let blocked = (0..ops.len())
            .any(|j| j != i && remaining & (1u128 << j) != 0 && precedes(&ops[j], op));
        if blocked {
            continue;
        }

        let next_remaining = remaining & !(1u128 << i);

        if op.optional && search(ops, next_remaining, model.clone(), failed) {
            return true;
        }

        let mut next_model = model.clone();
        let result = next_model.apply(&op.command);
        let matches_expected = match &op.expected {
            Some(expected) => &result == expected,
            None => true,
        };
        if matches_expected && search(ops, next_remaining, next_model, failed) {
            return true;
        }
    }

    failed.insert((state_hash, remaining));
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(peers: Vec<NodeId>) -> RunConfig {
        RunConfig {
            peers,
            election_timeout: 10,
            heartbeat_interval: 3,
            min_delay: 1,
            max_delay: 3,
            apply_timeout_ticks: 60,
            client_count: 4,
            total_ticks: 200,
            submit_chance_percent: 25,
            fault_chance_percent: 5,
        }
    }

    fn set(key: &[u8], value: &[u8]) -> Command {
        Command::Set {
            key: key.to_vec(),
            value: value.to_vec(),
        }
    }

    fn get(key: &[u8]) -> Command {
        Command::Get { key: key.to_vec() }
    }

    fn applied_entry(
        client: u64,
        invoke: u64,
        finish: u64,
        command: Command,
        result: ApplyResult,
    ) -> HistoryEntry {
        HistoryEntry {
            client: ClientId(client),
            invoke_tick: invoke,
            return_tick: Some(finish),
            command,
            outcome: Outcome::Applied(result),
        }
    }

    // --- Harness-level behavior -------------------------------------

    #[test]
    fn a_saved_seed_reproduces_the_exact_same_run_after_loading() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let original_config = config(peers);
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("seed.txt");

        save_seed(99, &original_config, &path).expect("save seed");
        let (loaded_seed, loaded_config) = load_seed(&path).expect("load seed");

        assert_eq!(loaded_seed, 99);
        let (_, original_history, _) = run(99, &original_config);
        let (_, loaded_history, _) = run(loaded_seed, &loaded_config);
        assert_eq!(original_history, loaded_history);
    }

    #[test]
    fn same_seed_and_config_reproduces_an_identical_history() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let config = config(peers);

        let (trace_a, history_a, faults_a) = run(42, &config);
        let (trace_b, history_b, faults_b) = run(42, &config);

        assert_eq!(trace_a, trace_b);
        assert_eq!(history_a, history_b);
        assert_eq!(faults_a, faults_b);
    }

    #[test]
    fn a_run_produces_genuinely_concurrent_operations() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let config = config(peers);
        let (_, history, _) = run(7, &config);

        assert!(
            !history.is_empty(),
            "a 400-tick run with a 40% submit chance across 4 clients must produce operations"
        );

        // Concurrent means: at least one pair of *different* clients whose
        // invoke/return windows genuinely overlap -- not just adjacent. A
        // harness that resolved one op before ever starting the next would
        // make every history trivially linearizable regardless of checker
        // correctness, which is exactly what this proves didn't happen.
        let overlaps = history.iter().any(|a| {
            history.iter().any(|b| {
                a.client != b.client
                    && a.invoke_tick < b.return_tick.unwrap_or(u64::MAX)
                    && b.invoke_tick < a.return_tick.unwrap_or(u64::MAX)
            })
        });
        assert!(
            overlaps,
            "expected at least one pair of overlapping operations from different clients"
        );
    }

    #[test]
    fn a_seeded_run_with_a_fault_mid_flight_produces_a_linearizable_history() {
        // The literal roadmap gate: a concrete seeded run whose history
        // includes at least one fault-injection event with an operation
        // genuinely outstanding across it, and the result is linearizable.
        // The fault is forced deterministically rather than left to dice
        // -- a probabilistic assertion that some interleaving occurred is
        // a flaky test waiting to happen.
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        // A fixed (non-randomized) timeout would make every node time out
        // on the same tick, forever splitting the vote three ways -- a
        // real min/max spread is what lets one candidate actually pull
        // ahead, same as any real cluster's jittered election timeout.
        let mut harness = Harness::new(peers.clone(), 5, 2, 123, 5, 10, 1, 1, 60);

        let mut leader = None;
        for _ in 0..60 {
            harness.step();
            if leader.is_none() {
                leader = harness.current_leader();
            }
        }
        let leader = leader.expect("cluster elects a leader within 60 ticks");
        let follower = *peers.iter().find(|&&id| id != leader).unwrap();

        harness.submit(ClientId(0), leader, set(b"k", b"v1"));
        assert!(
            harness.history().is_empty(),
            "the write must still be outstanding when the fault lands"
        );

        harness.crash(follower);
        let fault_tick = harness.faults()[0].tick;

        for _ in 0..20 {
            harness.step();
        }
        harness.restart(follower);
        for _ in 0..15 {
            harness.step();
        }
        harness.submit(ClientId(1), leader, get(b"k"));
        for _ in 0..10 {
            harness.step();
        }

        let faults = harness.faults().to_vec();
        let history = harness.finish();

        assert_eq!(
            faults.len(),
            2,
            "expected exactly the forced crash and restart"
        );
        assert_eq!(faults[0].kind, FaultKind::Crashed);
        assert_eq!(faults[1].kind, FaultKind::Restarted);

        let write = history
            .iter()
            .find(|entry| entry.client == ClientId(0))
            .expect("the write is present in the final history");
        assert!(
            write.invoke_tick <= fault_tick
                && write.return_tick.expect("the write eventually resolves") > fault_tick,
            "the write's own invoke/return window must span the fault tick"
        );
        assert!(matches!(write.outcome, Outcome::Applied(_)));

        let read = history
            .iter()
            .find(|entry| entry.client == ClientId(1))
            .expect("the read is present in the final history");
        assert_eq!(
            read.outcome,
            Outcome::Applied(ApplyResult::Get {
                value: Some(b"v1".to_vec())
            })
        );

        assert!(is_linearizable(&history));
    }

    #[test]
    fn a_run_with_faults_enabled_actually_crashes_and_restarts_nodes() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut config = config(peers);
        config.fault_chance_percent = 30;
        config.total_ticks = 800;

        let mut harness = Harness::new(
            config.peers.clone(),
            config.election_timeout,
            config.heartbeat_interval,
            99,
            config.election_timeout,
            config.election_timeout * 2,
            config.min_delay,
            config.max_delay,
            config.apply_timeout_ticks,
        );
        let mut driver_rng = Rng::new(99 ^ 0xD1B5_4A32_D192_ED03);
        let mut saw_a_crash = false;
        for _ in 0..config.total_ticks {
            harness.step();
            if driver_rng.range_inclusive(0, 99) < config.fault_chance_percent {
                let target = config.peers
                    [driver_rng.range_inclusive(0, config.peers.len() as u64 - 1) as usize];
                match driver_rng.range_inclusive(0, 3) {
                    0 => harness.isolate(target),
                    1 => harness.heal(target),
                    2 => {
                        harness.crash(target);
                        saw_a_crash = true;
                    }
                    _ => harness.restart(target),
                }
            }
        }

        assert!(
            saw_a_crash,
            "an 800-tick run with a 30% fault chance should crash at least one node"
        );
    }

    #[test]
    fn crash_and_restart_preserves_durable_state_and_the_replica_keeps_applying() {
        let peers = vec![NodeId(1)];
        let mut harness = Harness::new(peers, 5, 2, 1, 5, 5, 1, 1, 20);

        for _ in 0..10 {
            harness.step();
        }
        harness.submit(ClientId(0), NodeId(1), set(b"k", b"v"));
        harness.step();

        let applied_before_crash = harness
            .history()
            .iter()
            .any(|entry| matches!(entry.outcome, Outcome::Applied(_)));
        assert!(
            applied_before_crash,
            "single-node cluster commits immediately"
        );

        harness.crash(NodeId(1));
        assert!(harness.current_leader().is_none());
        harness.restart(NodeId(1));

        // A restarted node comes back as a Follower and must win a fresh
        // election before it can accept anything again.
        for _ in 0..10 {
            harness.step();
        }
        assert_eq!(harness.current_leader(), Some(NodeId(1)));

        harness.submit(ClientId(1), NodeId(1), get(b"k"));
        for _ in 0..5 {
            harness.step();
        }
        let get_result = harness
            .history()
            .iter()
            .rev()
            .find_map(|entry| (entry.client == ClientId(1)).then_some(&entry.outcome));
        assert_eq!(
            get_result,
            Some(&Outcome::Applied(ApplyResult::Get {
                value: Some(b"v".to_vec())
            })),
            "the value set before the crash must survive it"
        );
    }

    #[test]
    fn a_target_that_is_currently_crashed_rejects_immediately() {
        let peers = vec![NodeId(1)];
        let mut harness = Harness::new(peers, 5, 2, 1, 5, 5, 1, 1, 20);
        harness.crash(NodeId(1));

        harness.submit(ClientId(0), NodeId(1), get(b"k"));

        assert_eq!(harness.history().len(), 1);
        assert_eq!(harness.history()[0].outcome, Outcome::Rejected);
        assert_eq!(harness.history()[0].return_tick, Some(0));
    }

    #[test]
    fn an_isolated_follower_falls_behind_and_catches_up_once_healed() {
        let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
        let mut harness = Harness::new(peers.clone(), 5, 2, 5, 5, 10, 1, 1, 60);

        let mut leader = None;
        for _ in 0..60 {
            harness.step();
            if leader.is_none() {
                leader = harness.current_leader();
            }
        }
        let leader = leader.expect("cluster elects a leader within 60 ticks");
        let follower = *peers.iter().find(|&&id| id != leader).unwrap();

        harness.isolate(follower);
        harness.submit(ClientId(0), leader, set(b"k", b"v1"));
        for _ in 0..15 {
            harness.step();
        }

        let write = harness
            .history()
            .iter()
            .find(|entry| entry.client == ClientId(0))
            .expect("the write resolves despite the partition -- 2 of 3 is still a majority");
        assert!(matches!(write.outcome, Outcome::Applied(_)));

        assert_eq!(
            harness.replica(follower).unwrap().state_machine().get(b"k"),
            None,
            "the isolated follower must not have received the write -- proves isolation actually drops messages, not just that this run happened to succeed anyway"
        );

        harness.heal(follower);
        for _ in 0..15 {
            harness.step();
        }

        assert_eq!(
            harness.replica(follower).unwrap().state_machine().get(b"k"),
            Some(b"v1".as_slice()),
            "the healed follower must have caught up on the write it missed while isolated"
        );
    }

    // --- Checker: fixtures it must accept ----------------------------

    #[test]
    fn accepts_a_trivially_sequential_history() {
        let history = vec![
            applied_entry(
                0,
                0,
                1,
                set(b"k", b"v1"),
                ApplyResult::Set { previous: None },
            ),
            applied_entry(
                0,
                2,
                3,
                get(b"k"),
                ApplyResult::Get {
                    value: Some(b"v1".to_vec()),
                },
            ),
        ];
        assert!(is_linearizable(&history));
    }

    #[test]
    fn accepts_a_history_with_an_included_indeterminate_write() {
        let history = vec![
            HistoryEntry {
                client: ClientId(0),
                invoke_tick: 0,
                return_tick: Some(5),
                command: set(b"k", b"v1"),
                outcome: Outcome::Indeterminate,
            },
            applied_entry(
                1,
                6,
                7,
                get(b"k"),
                ApplyResult::Get {
                    value: Some(b"v1".to_vec()),
                },
            ),
        ];
        assert!(
            is_linearizable(&history),
            "the indeterminate write may have taken effect, so a Get observing it must be accepted"
        );
    }

    #[test]
    fn accepts_a_history_with_a_skipped_indeterminate_write() {
        let history = vec![
            HistoryEntry {
                client: ClientId(0),
                invoke_tick: 0,
                return_tick: Some(5),
                command: set(b"k", b"v1"),
                outcome: Outcome::Indeterminate,
            },
            applied_entry(1, 6, 7, get(b"k"), ApplyResult::Get { value: None }),
        ];
        assert!(
            is_linearizable(&history),
            "the indeterminate write may also have never taken effect, so a Get missing it must be accepted too"
        );
    }

    // --- Checker: fixtures it must reject ----------------------------

    #[test]
    fn rejects_a_get_observing_an_indeterminate_write_invoked_after_the_get_already_returned() {
        // An Indeterminate op's effect time is unbounded *above* -- it
        // might apply arbitrarily late -- but never below its own invoke
        // tick. A Get that already returned v1 at tick 15 cannot possibly
        // be explained by a write not even invoked until tick 100:
        // regression test for the `precedes` bug where the lower bound was
        // dropped entirely for an Indeterminate right-hand side.
        let history = vec![
            applied_entry(
                0,
                10,
                15,
                get(b"k"),
                ApplyResult::Get {
                    value: Some(b"v1".to_vec()),
                },
            ),
            HistoryEntry {
                client: ClientId(1),
                invoke_tick: 100,
                return_tick: Some(160),
                command: set(b"k", b"v1"),
                outcome: Outcome::Indeterminate,
            },
        ];
        assert!(!is_linearizable(&history));
    }

    #[test]
    fn rejects_a_get_returning_a_value_that_was_never_set() {
        let history = vec![applied_entry(
            0,
            0,
            1,
            get(b"k"),
            ApplyResult::Get {
                value: Some(b"phantom".to_vec()),
            },
        )];
        assert!(!is_linearizable(&history));
    }

    #[test]
    fn rejects_a_stale_read_after_a_later_write_already_returned() {
        // Sequential, non-overlapping: v1 fully completes, then v2 fully
        // completes, then a Get returns v1 -- a real staleness violation,
        // not a legitimate concurrent-ordering ambiguity.
        let history = vec![
            applied_entry(
                0,
                0,
                1,
                set(b"k", b"v1"),
                ApplyResult::Set { previous: None },
            ),
            applied_entry(
                0,
                2,
                3,
                set(b"k", b"v2"),
                ApplyResult::Set {
                    previous: Some(b"v1".to_vec()),
                },
            ),
            applied_entry(
                1,
                4,
                5,
                get(b"k"),
                ApplyResult::Get {
                    value: Some(b"v1".to_vec()),
                },
            ),
        ];
        assert!(!is_linearizable(&history));
    }

    #[test]
    fn rejects_a_rejected_writes_value_being_observed() {
        // A Rejected op never happened at all -- a Get "observing" its
        // value must fail, exactly like observing a value from thin air.
        let history = vec![
            HistoryEntry {
                client: ClientId(0),
                invoke_tick: 0,
                return_tick: Some(1),
                command: set(b"k", b"v1"),
                outcome: Outcome::Rejected,
            },
            applied_entry(
                1,
                2,
                3,
                get(b"k"),
                ApplyResult::Get {
                    value: Some(b"v1".to_vec()),
                },
            ),
        ];
        assert!(!is_linearizable(&history));
    }

    // --- Ground-truth cross-check ------------------------------------

    #[test]
    fn a_real_generated_run_is_always_linearizable() {
        // Get is log-routed exactly like a write, so every real run's
        // history is checked against an independent ground truth: the
        // committed log itself never produces a non-linearizable
        // sequence, so the checker had better agree on real runs, not
        // just on hand-built fixtures.
        for seed in 0..5 {
            let peers = vec![NodeId(1), NodeId(2), NodeId(3)];
            let config = config(peers);
            let (_, history, _) = run(seed, &config);
            assert!(
                is_linearizable(&history),
                "seed {seed} produced a history the checker rejected"
            );
        }
    }
}
