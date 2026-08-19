//! Proves the client gRPC server actually works over a real socket:
//! `RaftClientService`'s `Propose` handler wired to a real actor, both the
//! accepted and not-leader/redirect paths.

use std::net::SocketAddr;

use kurogane_kv::Replica;
use kurogane_raft::{Message, Node, NodeId};
use kurogane_runtime::actor::{self, Actor, PeerTransport};
use kurogane_runtime::proto::propose_reply;
use kurogane_runtime::proto::raft_client_client::RaftClientClient;
use kurogane_runtime::proto::raft_client_server::RaftClientServer;
use kurogane_runtime::proto::{Command, GetCommand, ProposeRequest};
use kurogane_runtime::server::RaftClientService;
use kurogane_runtime::storage::Storage;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;
use tonic::transport::Server;

struct NoopTransport;

impl PeerTransport for NoopTransport {
    fn send(&mut self, _to: NodeId, _message: Message) {}
}

/// Starts a real node behind a real TCP listener, serving only the client
/// service. Returns the address it's actually listening on.
async fn spawn_server(id: NodeId, peers: Vec<NodeId>) -> SocketAddr {
    let node = Node::new(id, peers, 1, 1).expect("valid node");
    let dir = tempdir().expect("temp dir");
    // Leaked deliberately: the tempdir must outlive the server task, and
    // this is a short-lived test process, not a long-running one.
    let storage_path = Box::leak(Box::new(dir)).path().join("state");
    let storage = Storage::open(storage_path).expect("open storage");
    let actor = Actor::new(Replica::new(node), storage, NoopTransport, u64::MAX);
    let (handle, receiver) = actor::channel(32);
    tokio::spawn(actor::run(actor, receiver));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let addr = listener.local_addr().expect("local addr");

    let service = RaftClientServer::new(RaftClientService::new(handle.clone()));
    tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("server must not fail");
    });

    // Drive the single-node cluster to leadership before returning, so
    // callers can propose immediately.
    handle.tick(1);
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    addr
}

#[tokio::test]
async fn a_propose_on_the_leader_round_trips_over_a_real_socket() {
    let addr = spawn_server(NodeId(1), vec![NodeId(1)]).await;

    let mut client = RaftClientClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to the real listener");

    let reply = client
        .propose(Request::new(ProposeRequest {
            command: Some(Command {
                kind: Some(kurogane_runtime::proto::command::Kind::Get(GetCommand {
                    key: vec![1],
                })),
            }),
        }))
        .await
        .expect("real RPC succeeds")
        .into_inner();

    match reply.result {
        Some(propose_reply::Result::Accepted(accepted)) => {
            assert_eq!(accepted.index, 1);
        }
        other => panic!("expected an accepted propose, got {other:?}"),
    }
}

#[tokio::test]
async fn a_propose_on_a_follower_returns_not_leader_over_a_real_socket() {
    let addr = spawn_server(NodeId(1), vec![NodeId(1), NodeId(2)]).await;

    let mut client = RaftClientClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to the real listener");

    let reply = client
        .propose(Request::new(ProposeRequest {
            command: Some(Command {
                kind: Some(kurogane_runtime::proto::command::Kind::Get(GetCommand {
                    key: vec![1],
                })),
            }),
        }))
        .await
        .expect("real RPC succeeds")
        .into_inner();

    match reply.result {
        Some(propose_reply::Result::NotLeader(not_leader)) => {
            assert_eq!(not_leader.leader_id, None);
        }
        other => panic!("expected a not-leader reply, got {other:?}"),
    }
}
