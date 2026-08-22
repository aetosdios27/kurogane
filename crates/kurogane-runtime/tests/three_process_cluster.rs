//! Real `kurogane-node` child processes, driven purely over gRPC (plus a
//! direct read of one node's on-disk storage file to confirm catch-up,
//! since there's no client-facing read API yet -- Get still round-trips
//! through the log same as any other command, but nothing surfaces its
//! applied value back over the wire in this milestone). Three gate tests:
//! durable writes replicate across real sockets, survive a leader kill and
//! a restart; (separately) a follower that's been stopped long enough for
//! the leader to compact past it recovers via a real InstallSnapshot RPC,
//! not by replaying history the leader no longer even has; and (separately)
//! a brand-new process joins as a learner via a real AddLearner RPC, gets
//! promoted to a full voter via a real ProposeConfigChange RPC once it's
//! actually caught up, and the resulting 4-voter cluster still makes
//! progress on 3-of-4 after losing one of its original members; (separately)
//! killing the leader that accepted a learner's AddLearner between
//! registration and promotion actually forces
//! add_learner_wait_catch_up_and_promote's re-registration retry to run
//! against a different leader, not just look correct by construction.

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kurogane_kv::{Command as KvCommand, StateMachine};
use kurogane_raft::LogPayload;
use kurogane_runtime::proto::raft_client_client::RaftClientClient;
use kurogane_runtime::proto::{
    AddLearnerRequest, Command as ProtoCommand, ProposeConfigChangeRequest, ProposeRequest,
    SetCommand, add_learner_reply, propose_config_change_reply, propose_reply,
};
use kurogane_runtime::storage::Storage;
use tempfile::TempDir;

const TOKEN: &str = "integration-test-token";
const ELECTION_TIMEOUT_TICKS: &str = "5";
const HEARTBEAT_INTERVAL_TICKS: &str = "1";
const TICK_INTERVAL_MS: &str = "50";
/// Effectively disables compaction for tests that don't exercise it --
/// nothing in those tests applies anywhere near this many entries.
const NO_COMPACTION: &str = "1000000";

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind an ephemeral port")
        .local_addr()
        .expect("local addr")
        .port()
}

/// Kills every child on drop, so a failed assertion (which unwinds the test
/// via panic) never leaks real OS processes.
struct ProcessGuard(Vec<Child>);

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn spawn_node(
    id: u64,
    peers: &str,
    client_addr: &str,
    storage_path: &std::path::Path,
    compact_threshold: &str,
    join_as_learner: bool,
) -> Child {
    let mut command = Command::new(env!("CARGO_BIN_EXE_kurogane-node"));
    command
        .env("KUROGANE_NODE_ID", id.to_string())
        .env("KUROGANE_PEERS", peers)
        .env("KUROGANE_CLIENT_ADDR", client_addr)
        .env("KUROGANE_STORAGE_PATH", storage_path)
        .env("KUROGANE_CLUSTER_TOKEN", TOKEN)
        .env("KUROGANE_ELECTION_TIMEOUT_TICKS", ELECTION_TIMEOUT_TICKS)
        .env(
            "KUROGANE_HEARTBEAT_INTERVAL_TICKS",
            HEARTBEAT_INTERVAL_TICKS,
        )
        .env("KUROGANE_TICK_INTERVAL_MS", TICK_INTERVAL_MS)
        .env("KUROGANE_COMPACT_THRESHOLD", compact_threshold)
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if join_as_learner {
        // Only meaningful on a genuinely fresh (never-before-used) storage
        // path -- see kurogane-node.rs's own is_fresh_storage guard. Every
        // caller of this branch in this file passes a brand-new tempdir.
        command.env("KUROGANE_JOIN_AS_LEARNER", "true");
    }
    command.spawn().expect("spawn kurogane-node")
}

/// Proposes `command` against whichever node in `client_addrs` turns out to
/// be leader, following redirect hints and retrying past processes that
/// aren't up yet or are mid-election, until `deadline`. Returns the
/// accepting node's id and the index it assigned.
async fn propose_via_any(
    client_addrs: &BTreeMap<u64, String>,
    command: ProtoCommand,
    deadline: Instant,
) -> (u64, u64) {
    let mut order: Vec<u64> = client_addrs.keys().copied().collect();
    loop {
        for id in order.clone() {
            if Instant::now() > deadline {
                panic!("no node accepted the propose before the deadline");
            }
            let addr = &client_addrs[&id];
            let Ok(mut client) = RaftClientClient::connect(format!("http://{addr}")).await else {
                continue;
            };
            let Ok(reply) = client
                .propose(ProposeRequest {
                    command: Some(command.clone()),
                })
                .await
            else {
                continue;
            };
            match reply.into_inner().result {
                Some(propose_reply::Result::Applied(applied)) => return (id, applied.index),
                Some(propose_reply::Result::NotLeader(not_leader)) => {
                    // A hint may point at a node that isn't a candidate for
                    // this call (e.g. a leader we've since killed, whose
                    // last-seen hint is still stale on a surviving node) --
                    // only reorder toward it if it's actually one we can
                    // try.
                    if let Some(hint) = not_leader.leader_id {
                        if client_addrs.contains_key(&hint) {
                            order.retain(|candidate| *candidate != hint);
                            order.insert(0, hint);
                        }
                    }
                }
                // A slow/uncertain apply on an otherwise-healthy leader --
                // this helper only cares about a definite outcome, so treat
                // it the same as no reply at all and retry.
                Some(propose_reply::Result::Indeterminate(_)) => {}
                None => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn set_command(key: &str, value: &str) -> ProtoCommand {
    ProtoCommand {
        kind: Some(kurogane_runtime::proto::command::Kind::Set(SetCommand {
            key: key.as_bytes().to_vec(),
            value: value.as_bytes().to_vec(),
        })),
    }
}

/// Reads `storage_path`'s current on-disk state and replays it into a
/// `StateMachine`, the same "read a node's on-disk storage file directly"
/// idiom the InstallSnapshot gate test above already uses, generalized into
/// a helper since the learner test below needs it at three different
/// points. Returns `None` -- meaning "not ready, try again" to every caller
/// -- rather than panicking on a decode error: `Storage::flush` rewrites
/// the file with a bare `fs::File::create` + `write_all`, not a
/// write-to-temp-then-rename, so a poll racing an in-flight write can
/// legitimately observe a torn/partial file. That is a timing artifact of
/// polling a live process's storage, not evidence of real corruption.
fn replicated_state(storage_path: &std::path::Path) -> Option<StateMachine> {
    let storage = Storage::open(storage_path).ok()?;
    let mut state = StateMachine::new();
    let boundary = storage.snapshot().last_included_index;
    if boundary > 0 {
        state.restore(boundary, storage.snapshot_data()).ok()?;
    }
    for entry in storage.log() {
        match &entry.payload {
            LogPayload::Command(bytes) => {
                let command = KvCommand::decode(bytes).ok()?;
                state.apply(&command);
            }
            LogPayload::Configuration(_) => {}
        }
    }
    Some(state)
}

/// True once `storage_path`'s persisted log contains a plain (non-joint)
/// `Configuration` entry whose voter set matches `expected_voters` exactly
/// (order-independent -- compared as sorted copies since the entry travels
/// through a repeated protobuf field on the wire and back). This, not
/// applied KV state, is the actual "the config change committed" signal:
/// per `Node::propose_config_change`'s own design, the leader appends the
/// plain `C_new` entry *only as a consequence of* the preceding joint
/// `C_old,new` entry having committed (the automatic joint -> plain
/// follow-up) -- so `C_new`'s mere presence in a node's persisted log is
/// causally downstream of that commit, unlike an ordinary command entry's
/// presence (which proves only replication, since a follower persists
/// `PersistLog` on acceptance, with no regard for `leader_commit`).
///
/// This scan requires the caller to run with compaction disabled: if
/// compaction fired, `C_new` could fold into `snapshot_config` and vanish
/// from `log()`, silently breaking the check. The learner test below uses
/// `NO_COMPACTION` for exactly this reason, not just to keep its log small.
fn has_committed_config(storage_path: &std::path::Path, expected_voters: &[u64]) -> bool {
    let Ok(storage) = Storage::open(storage_path) else {
        return false;
    };
    let mut expected: Vec<u64> = expected_voters.to_vec();
    expected.sort_unstable();
    storage.log().iter().any(|entry| match &entry.payload {
        LogPayload::Configuration(config) if config.old_voters.is_none() => {
            let mut voters: Vec<u64> = config.voters.iter().map(|id| id.0).collect();
            voters.sort_unstable();
            voters == expected
        }
        _ => false,
    })
}

/// Calls `AddLearner` against whichever node in `client_addrs` turns out to
/// be leader, following redirect hints exactly like `propose_via_any`.
/// Returns the accepting node's id -- the caller needs it, since promoting
/// this learner later must be pinned to that same leader (see the doc
/// comment on the promotion loop in the test below for why).
async fn add_learner_via_any(
    client_addrs: &BTreeMap<u64, String>,
    learner_id: u64,
    learner_peer_addr: &str,
    deadline: Instant,
) -> u64 {
    let mut order: Vec<u64> = client_addrs.keys().copied().collect();
    loop {
        for id in order.clone() {
            if Instant::now() > deadline {
                panic!("no node accepted AddLearner before the deadline");
            }
            let addr = &client_addrs[&id];
            let Ok(mut client) = RaftClientClient::connect(format!("http://{addr}")).await else {
                continue;
            };
            let Ok(reply) = client
                .add_learner(AddLearnerRequest {
                    node_id: learner_id,
                    address: learner_peer_addr.to_string(),
                })
                .await
            else {
                continue;
            };
            match reply.into_inner().result {
                Some(add_learner_reply::Result::Accepted(_)) => return id,
                Some(add_learner_reply::Result::NotLeader(not_leader)) => {
                    if let Some(hint) = not_leader.leader_id {
                        if client_addrs.contains_key(&hint) {
                            order.retain(|candidate| *candidate != hint);
                            order.insert(0, hint);
                        }
                    }
                }
                None => {}
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Registers `learner_id`/`learner_peer_addr` as a learner, waits for its
/// own storage to reflect every key/value pair in `expect_applied` (real
/// catch-up over real ticks, not assumed), then promotes it into
/// `new_voters` via `ProposeConfigChange` -- all three steps pinned to the
/// same leader.
///
/// Pinning matters: `AddLearner`'s effects (`Node::add_learner`'s
/// `PersistLearners`, and the transport's `add_peer` wiring) are entirely
/// leader-local -- learner registration is never itself replicated through
/// the log. If leadership moved during the catch-up wait, the new leader
/// has neither a learner record nor a transport address for `learner_id`,
/// and `propose_config_change` panics the next replication sweep the
/// moment it's asked to introduce a `NodeId` it has never seen (see that
/// function's own doc comment in kurogane-raft). So a `NotLeader` reply
/// here re-runs the entire sequence -- re-`AddLearner` (idempotent:
/// `Node::add_learner` is a no-op for an already-tracked id) against
/// whoever leads now, re-confirm catch-up, then retry the promotion --
/// rather than just retrying the one RPC.
async fn add_learner_wait_catch_up_and_promote(
    client_addrs: &BTreeMap<u64, String>,
    learner_id: u64,
    learner_peer_addr: &str,
    learner_storage_path: &std::path::Path,
    expect_applied: &[(&str, &str)],
    new_voters: Vec<u64>,
    deadline: Instant,
) -> u64 {
    loop {
        let leader =
            add_learner_via_any(client_addrs, learner_id, learner_peer_addr, deadline).await;

        loop {
            if let Some(state) = replicated_state(learner_storage_path) {
                let caught_up = expect_applied
                    .iter()
                    .all(|(key, value)| state.get(key.as_bytes()) == Some(value.as_bytes()));
                if caught_up {
                    break;
                }
            }
            assert!(
                Instant::now() < deadline,
                "learner should have caught up via ordinary replication before the deadline"
            );
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let addr = &client_addrs[&leader];
        let Ok(mut client) = RaftClientClient::connect(format!("http://{addr}")).await else {
            assert!(Instant::now() < deadline, "leader should remain reachable");
            continue;
        };
        let Ok(reply) = client
            .propose_config_change(ProposeConfigChangeRequest {
                voters: new_voters.clone(),
            })
            .await
        else {
            assert!(
                Instant::now() < deadline,
                "propose_config_change should have succeeded before the deadline"
            );
            continue;
        };
        match reply.into_inner().result {
            Some(propose_config_change_reply::Result::Accepted(_)) => return leader,
            // The leader that accepted AddLearner stepped down before we
            // could promote -- loop back to re-register the learner with
            // whoever's leading now, per this function's own doc comment.
            Some(propose_config_change_reply::Result::NotLeader(_)) | None => {}
        }
        assert!(
            Instant::now() < deadline,
            "config change should have been accepted before the deadline"
        );
    }
}

#[tokio::test]
async fn three_processes_replicate_survive_a_leader_kill_and_a_restarted_node_catches_up() {
    let ids = [1u64, 2, 3];
    let peer_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let client_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .collect::<Vec<_>>()
        .join(",");
    let client_addrs: BTreeMap<u64, String> = ids
        .iter()
        .map(|&id| (id, format!("127.0.0.1:{}", client_ports[&id])))
        .collect();

    let dirs: BTreeMap<u64, TempDir> = ids
        .iter()
        .map(|&id| (id, tempfile::tempdir().expect("temp dir")))
        .collect();
    let storage_paths: BTreeMap<u64, std::path::PathBuf> = dirs
        .iter()
        .map(|(&id, dir)| (id, dir.path().join("state")))
        .collect();

    let mut guard = ProcessGuard(
        ids.iter()
            .map(|&id| {
                spawn_node(
                    id,
                    &peers,
                    &client_addrs[&id],
                    &storage_paths[&id],
                    NO_COMPACTION,
                    false,
                )
            })
            .collect(),
    );

    let deadline = Instant::now() + Duration::from_secs(10);

    let (_, index) = propose_via_any(&client_addrs, set_command("k1", "v1"), deadline).await;
    assert_eq!(index, 1, "first write should land at log index 1");

    let (leader_id, index) =
        propose_via_any(&client_addrs, set_command("k2", "v2"), deadline).await;
    assert_eq!(index, 2, "second write should land at log index 2");

    // `Accepted` only means the leader assigned an index to its own log --
    // not that it's replicated to a majority yet. Give the heartbeat cycle
    // time to actually commit it before killing the leader, or losing it
    // would be correct Raft behavior (only committed entries survive a
    // leader change), not a bug to catch here.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let leader_index = ids.iter().position(|&id| id == leader_id).unwrap();
    guard.0[leader_index]
        .kill()
        .expect("kill the leader process");
    guard.0[leader_index].wait().expect("reap killed process");

    let remaining: BTreeMap<u64, String> = client_addrs
        .iter()
        .filter(|&(&id, _)| id != leader_id)
        .map(|(&id, addr)| (id, addr.clone()))
        .collect();

    let deadline = Instant::now() + Duration::from_secs(15);
    let (_, index) = propose_via_any(&remaining, set_command("k3", "v3"), deadline).await;
    assert_eq!(
        index, 3,
        "the surviving nodes must have retained entries 1-2 and continue the log, not reset it"
    );

    // Restart the killed node -- same id, same storage path, so it recovers
    // from disk and must catch up via replication from the current leader.
    guard.0[leader_index] = spawn_node(
        leader_id,
        &peers,
        &client_addrs[&leader_id],
        &storage_paths[&leader_id],
        NO_COMPACTION,
        false,
    );

    // Give it several heartbeat/tick rounds to reconnect and catch up.
    tokio::time::sleep(Duration::from_secs(2)).await;

    let recovered_log = Storage::open(&storage_paths[&leader_id])
        .expect("reopen the restarted node's storage")
        .log()
        .len();
    assert!(
        recovered_log >= 3,
        "restarted node should have caught up to at least 3 log entries, got {recovered_log}"
    );
}

#[tokio::test]
async fn a_stopped_follower_recovers_via_install_snapshot_after_the_leader_compacts_past_it() {
    const COMPACT_THRESHOLD: &str = "5";

    let ids = [1u64, 2, 3];
    let peer_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let client_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .collect::<Vec<_>>()
        .join(",");
    let client_addrs: BTreeMap<u64, String> = ids
        .iter()
        .map(|&id| (id, format!("127.0.0.1:{}", client_ports[&id])))
        .collect();

    let dirs: BTreeMap<u64, TempDir> = ids
        .iter()
        .map(|&id| (id, tempfile::tempdir().expect("temp dir")))
        .collect();
    let storage_paths: BTreeMap<u64, std::path::PathBuf> = dirs
        .iter()
        .map(|(&id, dir)| (id, dir.path().join("state")))
        .collect();

    let mut guard = ProcessGuard(
        ids.iter()
            .map(|&id| {
                spawn_node(
                    id,
                    &peers,
                    &client_addrs[&id],
                    &storage_paths[&id],
                    COMPACT_THRESHOLD,
                    false,
                )
            })
            .collect(),
    );

    let deadline = Instant::now() + Duration::from_secs(10);

    let (leader_id, _) = propose_via_any(&client_addrs, set_command("seed", "0"), deadline).await;

    // Stop a follower right away -- it will miss every write that
    // follows, and the leader's own next_index bookkeeping for it stays
    // frozen at wherever it last synced: an unreachable peer's send
    // attempts are simply dropped, never processed as a failure response,
    // so nothing ever backs next_index off further.
    let follower_id = *ids
        .iter()
        .find(|&&id| id != leader_id)
        .expect("a follower exists in a 3-node cluster");
    let follower_index = ids.iter().position(|&id| id == follower_id).unwrap();
    guard.0[follower_index]
        .kill()
        .expect("stop the follower process");
    guard.0[follower_index]
        .wait()
        .expect("reap stopped process");

    let remaining: BTreeMap<u64, String> = client_addrs
        .iter()
        .filter(|&(&id, _)| id != follower_id)
        .map(|(&id, addr)| (id, addr.clone()))
        .collect();

    // Well past COMPACT_THRESHOLD, so the leader compacts multiple times
    // while the follower is down -- its boundary ends up far beyond
    // wherever the stopped follower's next_index is frozen.
    for i in 0..20 {
        propose_via_any(
            &remaining,
            set_command(&format!("k{i}"), &format!("v{i}")),
            deadline,
        )
        .await;
    }

    // Restart the stopped follower -- same id, same (nearly empty)
    // storage path. Its next_index on the leader is now far behind the
    // leader's own boundary, so the very next heartbeat it receives must
    // be an InstallSnapshot, not an AppendEntries -- the leader no longer
    // even has those older entries to send.
    guard.0[follower_index] = spawn_node(
        follower_id,
        &peers,
        &client_addrs[&follower_id],
        &storage_paths[&follower_id],
        COMPACT_THRESHOLD,
        false,
    );

    // Poll rather than sleep-once: the leader still needs to actually
    // commit/apply/compact all 21 writes, and the follower still needs to
    // reconnect and receive the resulting snapshot, both of which can take
    // longer than any fixed guess under CI/parallel-test load.
    // The compaction trigger deliberately retains a trailing margin of the
    // leader's most recent entries out of the snapshot boundary, so a
    // briefly-behind peer can still catch up via ordinary AppendEntries --
    // meaning k18/k19 only land on the restarted follower through
    // replication *after* the snapshot install, not inside the snapshot
    // blob itself. Checking the blob alone is a race; the retained log
    // suffix (recovered.log()) has to be replayed on top of it too, and the
    // whole check has to stay inside the poll loop rather than a one-shot
    // break after the boundary first appears.
    let poll_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let recovered = Storage::open(&storage_paths[&follower_id])
            .expect("reopen the restarted follower's storage");
        let boundary = recovered.snapshot().last_included_index;
        if boundary > 0 {
            let mut state = StateMachine::new();
            state
                .restore(boundary, recovered.snapshot_data())
                .expect("the installed snapshot is well-formed");
            // The snapshot blob already reflects every key applied as of
            // the leader's snapshot instant, which is above `boundary` by
            // the retain margin, so this replay re-applies some entries
            // already present in the restored map. Harmless here (every
            // command is a Set on a distinct key, so reapplication is
            // idempotent) and not commit-index-gated (Storage doesn't
            // expose one) -- both fine for this single-sequential-writer
            // test, not a general guarantee.
            for entry in recovered.log() {
                match &entry.payload {
                    LogPayload::Command(bytes) => {
                        let command = KvCommand::decode(bytes)
                            .expect("this test only ever proposes commands it encoded itself");
                        state.apply(&command);
                    }
                    LogPayload::Configuration(_) => {}
                }
            }
            if state.get(b"k19") == Some(&b"v19"[..]) {
                break;
            }
        }
        assert!(
            Instant::now() < poll_deadline,
            "restarted follower should have installed a real snapshot and caught up on the \
             trailing entries via replication, not just replayed history (its log alone can't \
             explain catch-up, since the leader compacted those entries away)"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

#[tokio::test]
async fn a_learner_joins_gets_promoted_and_the_cluster_survives_losing_an_original_member() {
    let ids = [1u64, 2, 3];
    let learner_id = 4u64;

    let peer_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let client_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let learner_peer_port = free_port();
    let learner_client_port = free_port();

    // The three genesis processes' own KUROGANE_PEERS is untouched by the
    // 4th node's existence -- their transports only ever learn how to reach
    // it once a real AddLearner call lands on whichever of them is leader
    // (Actor::add_learner's GrpcPeerTransport::add_peer), never from their
    // own startup env.
    let peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .collect::<Vec<_>>()
        .join(",");
    // The learner's own KUROGANE_PEERS, by contrast, is repurposed
    // reachability, not membership (kurogane-node.rs's own module doc
    // comment): it needs its own entry (to bind its own peer listen
    // address) plus the three existing members' entries (so it can reach
    // them once it starts participating -- responding to their
    // AppendEntries, and later, once promoted, sending its own RequestVote/
    // AppendEntries traffic symmetrically).
    let learner_peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .chain(std::iter::once(format!(
            "{learner_id}=127.0.0.1:{learner_peer_port}"
        )))
        .collect::<Vec<_>>()
        .join(",");

    let client_addrs: BTreeMap<u64, String> = ids
        .iter()
        .map(|&id| (id, format!("127.0.0.1:{}", client_ports[&id])))
        .collect();
    let learner_client_addr = format!("127.0.0.1:{learner_client_port}");
    // AddLearnerRequest.address is a gRPC endpoint the leader's transport
    // connects to directly (Endpoint::from_shared), so it needs the scheme,
    // unlike KUROGANE_PEERS entries which the node binary prefixes itself.
    let learner_peer_addr = format!("http://127.0.0.1:{learner_peer_port}");

    let dirs: BTreeMap<u64, TempDir> = ids
        .iter()
        .map(|&id| (id, tempfile::tempdir().expect("temp dir")))
        .collect();
    let storage_paths: BTreeMap<u64, std::path::PathBuf> = dirs
        .iter()
        .map(|(&id, dir)| (id, dir.path().join("state")))
        .collect();
    // Held for the whole test -- dropping this would delete the learner's
    // storage file out from under its still-running process.
    let learner_dir = tempfile::tempdir().expect("temp dir");
    let learner_storage_path = learner_dir.path().join("state");

    let mut guard = ProcessGuard(
        ids.iter()
            .map(|&id| {
                spawn_node(
                    id,
                    &peers,
                    &client_addrs[&id],
                    &storage_paths[&id],
                    NO_COMPACTION,
                    false,
                )
            })
            .collect(),
    );

    let deadline = Instant::now() + Duration::from_secs(10);

    // Several writes, not one -- so the learner's later catch-up actually
    // exercises a real multi-entry replication backfill (starting from a
    // next_index seeded at the leader's current tip, per
    // Node::add_learner's doc comment, and backing off from there) instead
    // of a single two-round-trip no-op.
    let seed_pairs: [(&str, &str); 5] = [
        ("seed0", "v0"),
        ("seed1", "v1"),
        ("seed2", "v2"),
        ("seed3", "v3"),
        ("seed4", "v4"),
    ];
    let mut first_index = None;
    for (key, value) in seed_pairs {
        let (_, index) = propose_via_any(&client_addrs, set_command(key, value), deadline).await;
        first_index.get_or_insert(index);
    }
    assert_eq!(
        first_index,
        Some(1),
        "the very first write on a fresh cluster should land at log index 1 -- C_0 is bootstrap \
         state, not a log entry"
    );

    // Spawn the 4th process in join-mode against fresh storage: no prior
    // hard state/log/snapshot, so kurogane-node.rs's is_fresh_storage check
    // takes the Node::new_learner branch rather than Node::recover.
    guard.0.push(spawn_node(
        learner_id,
        &learner_peers,
        &learner_client_addr,
        &learner_storage_path,
        NO_COMPACTION,
        true,
    ));
    let all_ids = [1u64, 2, 3, learner_id];

    let new_voters = vec![1u64, 2, 3, learner_id];
    let promote_deadline = Instant::now() + Duration::from_secs(20);
    let _promoting_leader = add_learner_wait_catch_up_and_promote(
        &client_addrs,
        learner_id,
        &learner_peer_addr,
        &learner_storage_path,
        &seed_pairs,
        new_voters.clone(),
        promote_deadline,
    )
    .await;

    // Wait for the actual "config change committed" signal -- see
    // has_committed_config's own doc comment for why a plain C_new entry's
    // presence, not applied KV state, is what proves this -- to reach every
    // member's persisted log, original and newly promoted alike, before
    // treating the cluster as safely at 4 voters. Only once this holds is
    // losing one of the four (a 3-of-4 majority) a safe next step.
    let config_commit_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let all_committed = all_ids.iter().all(|id| {
            let path = if *id == learner_id {
                &learner_storage_path
            } else {
                &storage_paths[id]
            };
            has_committed_config(path, &new_voters)
        });
        if all_committed {
            break;
        }
        assert!(
            Instant::now() < config_commit_deadline,
            "the promotion's plain C_new config entry should have replicated to all four \
             members before the deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let mut all_client_addrs = client_addrs.clone();
    all_client_addrs.insert(learner_id, learner_client_addr.clone());

    // A write proposed after promotion, routed through whichever node is
    // leader now (which may or may not still be `promoting_leader`) --
    // confirms the 4-voter cluster, including the freshly promoted member,
    // actually replicates ordinary traffic end to end, not just the
    // membership-change entries themselves.
    let deadline = Instant::now() + Duration::from_secs(10);
    let (leader_after_promotion, _) = propose_via_any(
        &all_client_addrs,
        set_command("post-promotion", "confirmed"),
        deadline,
    )
    .await;

    let replicate_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let all_replicated = all_ids.iter().all(|id| {
            let path = if *id == learner_id {
                &learner_storage_path
            } else {
                &storage_paths[id]
            };
            replicated_state(path)
                .map(|state| state.get(b"post-promotion") == Some(b"confirmed"))
                .unwrap_or(false)
        });
        if all_replicated {
            break;
        }
        assert!(
            Instant::now() < replicate_deadline,
            "the post-promotion write should have replicated to all four members, including \
             the newly promoted one, before the deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Kill one of the *original* three members -- deliberately not the
    // (best-effort-known) current leader, and deliberately not the newly
    // promoted 4th node. A leader-kill scenario is already this file's
    // first gate test; what this test specifically needs to prove is that
    // the cluster survives losing an *original* member once a new one has
    // joined and been promoted, which is a distinct claim about the
    // now-4-member configuration, not about leader failover. Avoiding
    // `leader_after_promotion` is still only best-effort -- leadership can
    // move again in the gap before the kill actually lands -- but that's
    // fine either way: 3-of-4 is a majority regardless of which specific
    // member is lost.
    let victim = *ids
        .iter()
        .find(|&&id| id != leader_after_promotion)
        .expect("at least one of the three original members isn't the leader");
    let victim_index = all_ids.iter().position(|&id| id == victim).unwrap();
    guard.0[victim_index]
        .kill()
        .expect("kill the victim process");
    guard.0[victim_index].wait().expect("reap killed process");

    let remaining_client_addrs: BTreeMap<u64, String> = all_client_addrs
        .iter()
        .filter(|&(&id, _)| id != victim)
        .map(|(&id, addr)| (id, addr.clone()))
        .collect();

    // The cluster is now 3 live members out of 4 configured voters -- still
    // a majority. Confirm it keeps making progress: propose a new write
    // through whichever of the three survivors is now leader (it may well
    // have changed) and confirm it actually replicates to all three
    // survivors, including the promoted 4th node carrying real load, not
    // just the two remaining original members.
    let deadline = Instant::now() + Duration::from_secs(15);
    propose_via_any(
        &remaining_client_addrs,
        set_command("post-kill", "still-alive"),
        deadline,
    )
    .await;

    let final_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let all_survivors_caught_up = all_ids.iter().filter(|&&id| id != victim).all(|id| {
            let path = if *id == learner_id {
                &learner_storage_path
            } else {
                &storage_paths[id]
            };
            replicated_state(path)
                .map(|state| state.get(b"post-kill") == Some(b"still-alive"))
                .unwrap_or(false)
        });
        if all_survivors_caught_up {
            break;
        }
        assert!(
            Instant::now() < final_deadline,
            "the surviving 3-of-4 members (two original plus the promoted learner) should still \
             make progress and replicate a new write after losing one original member"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Proves the retry loop inside `add_learner_wait_catch_up_and_promote`
/// (documented on that function as handling "the leader that accepted
/// AddLearner stepped down before we could promote") actually executes
/// against a real second leader, not just that it's correct by construction.
/// Registers the learner against one leader, kills that exact process, then
/// drives promotion against the survivors -- which can only succeed by
/// re-`AddLearner`ing against whichever of them leads now, since the killed
/// leader's `PersistLearners`/transport wiring for this learner is gone with
/// it (leader-local state, per `add_learner_wait_catch_up_and_promote`'s own
/// doc comment). The discriminating assertion is `assert_ne!(promoting_leader,
/// original_leader)`: a test that only checked promotion eventually succeeds
/// wouldn't tell "the retry path ran" apart from "promotion succeeded some
/// other way."
#[tokio::test]
async fn add_learner_promotion_retries_after_the_registering_leader_is_killed() {
    let ids = [1u64, 2, 3];
    let learner_id = 4u64;

    let peer_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let client_ports: BTreeMap<u64, u16> = ids.iter().map(|&id| (id, free_port())).collect();
    let learner_peer_port = free_port();
    let learner_client_port = free_port();

    let peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .collect::<Vec<_>>()
        .join(",");
    let learner_peers = ids
        .iter()
        .map(|id| format!("{id}=127.0.0.1:{}", peer_ports[id]))
        .chain(std::iter::once(format!(
            "{learner_id}=127.0.0.1:{learner_peer_port}"
        )))
        .collect::<Vec<_>>()
        .join(",");

    let client_addrs: BTreeMap<u64, String> = ids
        .iter()
        .map(|&id| (id, format!("127.0.0.1:{}", client_ports[&id])))
        .collect();
    let learner_client_addr = format!("127.0.0.1:{learner_client_port}");
    let learner_peer_addr = format!("http://127.0.0.1:{learner_peer_port}");

    let dirs: BTreeMap<u64, TempDir> = ids
        .iter()
        .map(|&id| (id, tempfile::tempdir().expect("temp dir")))
        .collect();
    let storage_paths: BTreeMap<u64, std::path::PathBuf> = dirs
        .iter()
        .map(|(&id, dir)| (id, dir.path().join("state")))
        .collect();
    let learner_dir = tempfile::tempdir().expect("temp dir");
    let learner_storage_path = learner_dir.path().join("state");

    let mut guard = ProcessGuard(
        ids.iter()
            .map(|&id| {
                spawn_node(
                    id,
                    &peers,
                    &client_addrs[&id],
                    &storage_paths[&id],
                    NO_COMPACTION,
                    false,
                )
            })
            .collect(),
    );

    let deadline = Instant::now() + Duration::from_secs(10);

    let seed_pairs: [(&str, &str); 3] = [("seed0", "v0"), ("seed1", "v1"), ("seed2", "v2")];
    for (key, value) in seed_pairs {
        propose_via_any(&client_addrs, set_command(key, value), deadline).await;
    }

    guard.0.push(spawn_node(
        learner_id,
        &learner_peers,
        &learner_client_addr,
        &learner_storage_path,
        NO_COMPACTION,
        true,
    ));
    let all_ids = [1u64, 2, 3, learner_id];
    let new_voters = vec![1u64, 2, 3, learner_id];

    // Register the learner directly (not via the combined helper) so this
    // test can capture and then kill exactly the leader that accepted it,
    // rather than an arbitrary/best-effort guess at current leadership.
    let register_deadline = Instant::now() + Duration::from_secs(10);
    let original_leader = add_learner_via_any(
        &client_addrs,
        learner_id,
        &learner_peer_addr,
        register_deadline,
    )
    .await;

    let original_leader_index = ids.iter().position(|&id| id == original_leader).unwrap();
    guard.0[original_leader_index]
        .kill()
        .expect("kill the registering leader process");
    guard.0[original_leader_index]
        .wait()
        .expect("reap killed process");

    let surviving_client_addrs: BTreeMap<u64, String> = client_addrs
        .iter()
        .filter(|&(&id, _)| id != original_leader)
        .map(|(&id, addr)| (id, addr.clone()))
        .collect();

    // Promotion can only succeed here by re-AddLearner-ing against one of
    // the two surviving original members -- the killed leader's learner
    // record and transport wiring for this learner died with it.
    let promote_deadline = Instant::now() + Duration::from_secs(20);
    let promoting_leader = add_learner_wait_catch_up_and_promote(
        &surviving_client_addrs,
        learner_id,
        &learner_peer_addr,
        &learner_storage_path,
        &seed_pairs,
        new_voters.clone(),
        promote_deadline,
    )
    .await;

    assert_ne!(
        promoting_leader, original_leader,
        "promotion must have gone through a different leader than the one that registered the \
         learner and was then killed -- proving the re-registration retry path actually ran, \
         not just that promotion eventually succeeded some other way"
    );

    let config_commit_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let all_committed = all_ids
            .iter()
            .filter(|&&id| id != original_leader)
            .all(|id| {
                let path = if *id == learner_id {
                    &learner_storage_path
                } else {
                    &storage_paths[id]
                };
                has_committed_config(path, &new_voters)
            });
        if all_committed {
            break;
        }
        assert!(
            Instant::now() < config_commit_deadline,
            "the promotion's plain C_new config entry should have replicated to every surviving \
             member before the deadline"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    let mut all_client_addrs = surviving_client_addrs.clone();
    all_client_addrs.insert(learner_id, learner_client_addr.clone());

    let deadline = Instant::now() + Duration::from_secs(10);
    propose_via_any(
        &all_client_addrs,
        set_command("post-promotion", "confirmed"),
        deadline,
    )
    .await;

    let replicate_deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let all_replicated = all_ids
            .iter()
            .filter(|&&id| id != original_leader)
            .all(|id| {
                let path = if *id == learner_id {
                    &learner_storage_path
                } else {
                    &storage_paths[id]
                };
                replicated_state(path)
                    .map(|state| state.get(b"post-promotion") == Some(b"confirmed"))
                    .unwrap_or(false)
            });
        if all_replicated {
            break;
        }
        assert!(
            Instant::now() < replicate_deadline,
            "the post-promotion write should have replicated to every surviving member, \
             including the newly promoted learner"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}
