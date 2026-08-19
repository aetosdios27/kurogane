//! Real `kurogane-node` child processes, driven purely over gRPC (plus a
//! direct read of one node's on-disk storage file to confirm catch-up,
//! since there's no client-facing read API yet -- Get still round-trips
//! through the log same as any other command, but nothing surfaces its
//! applied value back over the wire in this milestone). Two gate tests:
//! durable writes replicate across real sockets, survive a leader kill and
//! a restart, and (separately) a follower that's been stopped long enough
//! for the leader to compact past it recovers via a real InstallSnapshot
//! RPC, not by replaying history the leader no longer even has.

use std::collections::BTreeMap;
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use kurogane_kv::StateMachine;
use kurogane_runtime::proto::raft_client_client::RaftClientClient;
use kurogane_runtime::proto::{Command as ProtoCommand, ProposeRequest, SetCommand, propose_reply};
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
) -> Child {
    Command::new(env!("CARGO_BIN_EXE_kurogane-node"))
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
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn kurogane-node")
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
                Some(propose_reply::Result::Accepted(accepted)) => return (id, accepted.index),
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
    );

    // Poll rather than sleep-once: the leader still needs to actually
    // commit/apply/compact all 21 writes, and the follower still needs to
    // reconnect and receive the resulting snapshot, both of which can take
    // longer than any fixed guess under CI/parallel-test load.
    let poll_deadline = Instant::now() + Duration::from_secs(15);
    let (boundary, recovered) = loop {
        let recovered = Storage::open(&storage_paths[&follower_id])
            .expect("reopen the restarted follower's storage");
        let boundary = recovered.snapshot().last_included_index;
        if boundary > 0 {
            break (boundary, recovered);
        }
        assert!(
            Instant::now() < poll_deadline,
            "restarted follower should have installed a real snapshot, not just replayed history \
             (its log alone can't explain catch-up, since the leader compacted those entries away)"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };

    let mut state = StateMachine::new();
    state
        .restore(boundary, recovered.snapshot_data())
        .expect("the installed snapshot is well-formed");
    assert_eq!(
        state.get(b"k19"),
        Some(&b"v19"[..]),
        "the installed snapshot must reflect writes made after the follower was stopped"
    );
}
