//! Process entry point: loads config from the environment, wires up
//! storage, the actor, the wall-clock timer, and both gRPC servers, then
//! runs until killed. All configuration is environment variables — this is
//! a learning project's dev/test binary, not a deployed service, so a
//! small hand-rolled parser is enough and avoids pulling in a CLI-args
//! dependency.

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

use kurogane_kv::Replica;
use kurogane_raft::{ClusterConfig, Node, NodeId, Snapshot};
use kurogane_runtime::actor::{self, Actor};
use kurogane_runtime::auth::TokenInterceptor;
use kurogane_runtime::peer_client::GrpcPeerTransport;
use kurogane_runtime::proto::raft_client_server::RaftClientServer;
use kurogane_runtime::proto::raft_peer_server::RaftPeerServer;
use kurogane_runtime::server::{RaftClientService, RaftPeerService};
use kurogane_runtime::storage::Storage;
use kurogane_runtime::timer;
use tonic::transport::Server;

/// `id=host:port` pairs for every cluster member, including this node —
/// this node's own listen address for the peer service is read back out of
/// its own entry, so it's never configured twice.
fn parse_peers(raw: &str) -> BTreeMap<NodeId, String> {
    raw.split(',')
        .map(|entry| {
            let (id, address) = entry
                .split_once('=')
                .unwrap_or_else(|| panic!("KUROGANE_PEERS entry '{entry}' is missing '='"));
            let id: u64 = id
                .parse()
                .unwrap_or_else(|_| panic!("KUROGANE_PEERS entry '{entry}' has a non-numeric id"));
            (NodeId(id), address.to_string())
        })
        .collect()
}

fn env_var(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"))
}

fn env_var_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| panic!("{name} is not a valid value"))
        })
        .unwrap_or(default)
}

#[tokio::main]
async fn main() {
    let id = NodeId(
        env_var("KUROGANE_NODE_ID")
            .parse()
            .expect("KUROGANE_NODE_ID is a valid u64"),
    );
    let peers = parse_peers(&env_var("KUROGANE_PEERS"));
    let member_ids: Vec<NodeId> = peers.keys().copied().collect();
    let peer_listen_addr: SocketAddr = peers
        .get(&id)
        .unwrap_or_else(|| panic!("KUROGANE_PEERS has no entry for KUROGANE_NODE_ID={}", id.0))
        .parse()
        .expect("this node's own KUROGANE_PEERS entry is a valid socket address");
    let client_listen_addr: SocketAddr = env_var("KUROGANE_CLIENT_ADDR")
        .parse()
        .expect("KUROGANE_CLIENT_ADDR is a valid socket address");
    let storage_path = env_var("KUROGANE_STORAGE_PATH");
    let token = env_var("KUROGANE_CLUSTER_TOKEN");
    let election_timeout_ticks: u64 = env_var_or("KUROGANE_ELECTION_TIMEOUT_TICKS", 10);
    let heartbeat_interval_ticks: u64 = env_var_or("KUROGANE_HEARTBEAT_INTERVAL_TICKS", 2);
    let tick_interval_ms: u64 = env_var_or("KUROGANE_TICK_INTERVAL_MS", 50);
    let compact_threshold: u64 = env_var_or("KUROGANE_COMPACT_THRESHOLD", 50);

    let storage = Storage::open(storage_path).expect("open durable storage");
    let node = Node::recover(
        id,
        member_ids,
        election_timeout_ticks,
        heartbeat_interval_ticks,
        storage.hard_state(),
        storage.log().to_vec(),
        Snapshot {
            metadata: storage.snapshot(),
            data: storage.snapshot_data().to_vec(),
            // Storage doesn't yet round-trip a persisted snapshot config --
            // durable snapshot-config handling in Storage/StorageState is
            // separate, later cross-crate work, not this stage's job.
            config: ClusterConfig::default(),
        },
        // Storage doesn't yet round-trip a persisted learner set --
        // durable PersistLearners handling in Storage/StorageState is
        // separate, later cross-crate work, not this stage's job.
        Vec::new(),
    )
    .expect("valid node configuration");

    let (handle, receiver) = actor::channel(64);

    let peer_addresses: BTreeMap<NodeId, String> = peers
        .into_iter()
        .filter(|(peer_id, _)| *peer_id != id)
        .map(|(peer_id, address)| (peer_id, format!("http://{address}")))
        .collect();
    let transport = GrpcPeerTransport::new(peer_addresses, token.clone(), handle.clone());

    let actor = Actor::new(
        Replica::recover(node),
        storage,
        transport,
        compact_threshold,
    );
    tokio::spawn(actor::run(actor, receiver));

    tokio::spawn(timer::run(
        handle.clone(),
        Duration::from_millis(tick_interval_ms),
        election_timeout_ticks,
        election_timeout_ticks * 2,
    ));

    let peer_service = RaftPeerServer::with_interceptor(
        RaftPeerService::new(handle.clone()),
        TokenInterceptor::new(token),
    );
    tokio::spawn(async move {
        Server::builder()
            .add_service(peer_service)
            .serve(peer_listen_addr)
            .await
            .expect("peer gRPC server must not fail");
    });

    let client_service = RaftClientServer::new(RaftClientService::new(handle));
    eprintln!(
        "kurogane-node {}: peer={peer_listen_addr} client={client_listen_addr}",
        id.0
    );
    Server::builder()
        .add_service(client_service)
        .serve(client_listen_addr)
        .await
        .expect("client gRPC server must not fail");
}
