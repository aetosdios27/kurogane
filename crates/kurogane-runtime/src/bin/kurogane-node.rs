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
use kurogane_raft::{HardState, Node, NodeId, Snapshot};
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
    // Set (any value works; a simple presence-style flag, same idiom this
    // file already uses via env_var_or's generic parse -- bool's own
    // FromStr accepts exactly "true"/"false") on a brand-new node that's
    // joining an already-running cluster rather than bootstrapping one from
    // scratch. See the Node::new_learner branch below for exactly what this
    // changes and doesn't.
    let join_as_learner: bool = env_var_or("KUROGANE_JOIN_AS_LEARNER", false);

    let storage = Storage::open(storage_path).expect("open durable storage");

    // "Fresh" here means this node has never actually recovered anything
    // real yet -- no hard state, no log, no snapshot. Only then is it safe
    // to start via Node::new_learner: a *restart* of an already-admitted
    // learner/voter (KUROGANE_JOIN_AS_LEARNER left set across restarts, or
    // simply still set out of caution) must fall through to the ordinary
    // Node::recover path below instead, since its real snapshot/log by now
    // carries the actual membership -- Node::recover already knows to
    // prefer that over anything this env var could say.
    let is_fresh_storage = storage.hard_state() == HardState::default()
        && storage.log().is_empty()
        && storage.snapshot().last_included_index == 0;

    let node = if join_as_learner && is_fresh_storage {
        Node::new_learner(id, election_timeout_ticks, heartbeat_interval_ticks)
            .expect("valid node configuration")
    } else {
        // member_ids (from KUROGANE_PEERS) still matters here even in join
        // mode that fell through to this branch on a restart: recover only
        // ever *falls back* to wrapping it as the bootstrap (C_0) config
        // when there's truly no prior snapshot at all, which can't be true
        // once is_fresh_storage was false.
        //
        // Deliberate simplification for a genesis (non-join) node too: this
        // is a learning project's dev/test binary, not a deployed service
        // (see the module doc comment), so KUROGANE_PEERS is still required
        // even when joining -- it's just repurposed. A join-mode node's
        // *membership* comes entirely from Node::new_learner's empty config
        // and later real AddLearner/ProposeConfigChange traffic, never from
        // this list; KUROGANE_PEERS here only ever seeds peer_addresses/
        // GrpcPeerTransport below (so this node knows how to reach the rest
        // of the cluster, and how the rest of the cluster's own
        // KUROGANE_PEERS entry for this node's id resolves its own listen
        // address) -- reachability, not membership.
        Node::recover(
            id,
            member_ids,
            election_timeout_ticks,
            heartbeat_interval_ticks,
            storage.hard_state(),
            storage.log().to_vec(),
            Snapshot {
                metadata: storage.snapshot(),
                data: storage.snapshot_data().to_vec(),
                config: storage.snapshot_config().clone(),
            },
            storage.learners().to_vec(),
        )
        .expect("valid node configuration")
    };

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
