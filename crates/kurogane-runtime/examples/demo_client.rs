//! Manual demo driver, not part of the crate's public surface -- a scratch
//! tool for watching a real cluster behave by hand (leader redirection,
//! replication, compaction/InstallSnapshot), mirroring the throwaway
//! smoke_client.rs used to verify milestone six the same way.
//!
//! Usage:
//!   demo_client propose <http://addr> set <key> <value>
//!   demo_client propose <http://addr> get <key>
//!   demo_client propose <http://addr> delete <key>
//!   demo_client inspect <storage-path> [key]

use std::env;

use kurogane_kv::{Command as KvCommand, StateMachine};
use kurogane_raft::LogPayload;
use kurogane_runtime::proto::raft_client_client::RaftClientClient;
use kurogane_runtime::proto::{
    Command, DeleteCommand, GetCommand, ProposeRequest, SetCommand, command, propose_reply,
};
use kurogane_runtime::storage::Storage;

fn usage() -> ! {
    eprintln!(
        "usage:\n  demo_client propose <http://addr> set <key> <value>\n  demo_client propose <http://addr> get <key>\n  demo_client propose <http://addr> delete <key>\n  demo_client inspect <storage-path> [key]"
    );
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("propose") => propose(&args[1..]).await,
        Some("inspect") => inspect(&args[1..]),
        _ => usage(),
    }
}

async fn propose(args: &[String]) {
    let [addr, verb, rest @ ..] = args else {
        usage()
    };
    let command = match (verb.as_str(), rest) {
        ("set", [key, value]) => command::Kind::Set(SetCommand {
            key: key.clone().into_bytes(),
            value: value.clone().into_bytes(),
        }),
        ("get", [key]) => command::Kind::Get(GetCommand {
            key: key.clone().into_bytes(),
        }),
        ("delete", [key]) => command::Kind::Delete(DeleteCommand {
            key: key.clone().into_bytes(),
        }),
        _ => usage(),
    };

    let mut client = RaftClientClient::connect(addr.clone())
        .await
        .unwrap_or_else(|error| panic!("connect to {addr}: {error}"));
    let reply = client
        .propose(ProposeRequest {
            command: Some(Command {
                kind: Some(command),
            }),
        })
        .await
        .unwrap_or_else(|status| panic!("propose RPC failed: {status}"))
        .into_inner();

    match reply.result {
        Some(propose_reply::Result::Applied(applied)) => {
            println!("applied at index {}: {:?}", applied.index, applied.result);
        }
        Some(propose_reply::Result::NotLeader(not_leader)) => {
            println!("not leader; hint = {:?}", not_leader.leader_id);
        }
        Some(propose_reply::Result::Indeterminate(_)) => {
            println!("indeterminate: may or may not have applied");
        }
        None => println!("empty reply"),
    }
}

/// Reconstructs applied state from storage directly, the same way the
/// InstallSnapshot integration test does -- there's no client-facing read
/// RPC yet, so this is the only way to see what a node actually holds.
fn inspect(args: &[String]) {
    let (path, key) = match args {
        [path] => (path, None),
        [path, key] => (path, Some(key)),
        _ => usage(),
    };
    let storage = Storage::open(path).unwrap_or_else(|error| panic!("open {path}: {error}"));
    let snapshot = storage.snapshot();

    println!("hard_state: {:?}", storage.hard_state());
    println!("log: {} entries", storage.log().len());
    println!(
        "snapshot boundary: last_included_index={} last_included_term={} ({} bytes)",
        snapshot.last_included_index,
        snapshot.last_included_term,
        storage.snapshot_data().len()
    );

    if let Some(key) = key {
        let mut state = StateMachine::new();
        if snapshot.last_included_index > 0 {
            state
                .restore(snapshot.last_included_index, storage.snapshot_data())
                .expect("well-formed snapshot");
        }
        for entry in storage.log() {
            match &entry.payload {
                LogPayload::Command(bytes) => {
                    let command = KvCommand::decode(bytes)
                        .expect("this driver only ever proposes commands it encoded itself");
                    state.apply(&command);
                }
                LogPayload::Configuration(_) => {}
            }
        }
        println!("{key} = {:?}", state.get(key.as_bytes()));
    }
}
