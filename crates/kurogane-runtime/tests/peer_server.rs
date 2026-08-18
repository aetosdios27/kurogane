//! Proves the peer gRPC server actually works over a real socket: the
//! token interceptor, `RaftPeerService`, and the actor wired together.
//! `GrpcPeerTransport`'s outbound half gets its own real-network exercise
//! as part of the full multi-process integration test.

use std::net::SocketAddr;

use kurogane_kv::Replica;
use kurogane_raft::{Message, Node, NodeId};
use kurogane_runtime::actor::{self, Actor, PeerTransport};
use kurogane_runtime::auth::{TokenInterceptor, attach_token};
use kurogane_runtime::proto::RequestVoteRequest;
use kurogane_runtime::proto::raft_peer_client::RaftPeerClient;
use kurogane_runtime::proto::raft_peer_server::RaftPeerServer;
use kurogane_runtime::server::RaftPeerService;
use kurogane_runtime::storage::Storage;
use tempfile::tempdir;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::Request;
use tonic::transport::Server;

const TOKEN: &str = "integration-test-token";

struct NoopTransport;

impl PeerTransport for NoopTransport {
    fn send(&mut self, _to: NodeId, _message: Message) {}
}

/// Starts a real node behind a real TCP listener: an `Actor` running in its
/// own task, served over gRPC with the token interceptor attached. Returns
/// the address it's actually listening on.
async fn spawn_server(id: NodeId, peers: Vec<NodeId>) -> SocketAddr {
    let node = Node::new(id, peers, 5, 2).expect("valid node");
    let dir = tempdir().expect("temp dir");
    // Leaked deliberately: the tempdir must outlive the server task, and
    // this is a short-lived test process, not a long-running one.
    let storage_path = Box::leak(Box::new(dir)).path().join("state");
    let storage = Storage::open(storage_path).expect("open storage");
    let actor = Actor::new(Replica::new(node), storage, NoopTransport);
    let (handle, receiver) = actor::channel(32);
    tokio::spawn(actor::run(actor, receiver));

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let addr = listener.local_addr().expect("local addr");

    let service = RaftPeerServer::with_interceptor(
        RaftPeerService::new(handle),
        TokenInterceptor::new(TOKEN),
    );
    tokio::spawn(async move {
        Server::builder()
            .add_service(service)
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .expect("server must not fail");
    });

    addr
}

#[tokio::test]
async fn a_request_vote_round_trips_over_a_real_socket() {
    let addr = spawn_server(NodeId(1), vec![NodeId(1), NodeId(2)]).await;

    let mut client = RaftPeerClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to the real listener");

    let request = attach_token(
        Request::new(RequestVoteRequest {
            term: 1,
            candidate_id: 2,
            last_log_index: 0,
            last_log_term: 0,
        }),
        TOKEN,
    );
    let reply = client
        .request_vote(request)
        .await
        .expect("real RPC succeeds")
        .into_inner();

    assert_eq!(reply.term, 1);
    assert!(reply.granted);
}

#[tokio::test]
async fn a_request_without_the_cluster_token_is_rejected_over_a_real_socket() {
    let addr = spawn_server(NodeId(1), vec![NodeId(1), NodeId(2)]).await;

    let mut client = RaftPeerClient::connect(format!("http://{addr}"))
        .await
        .expect("connect to the real listener");

    let request = Request::new(RequestVoteRequest {
        term: 1,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    });
    let error = client
        .request_vote(request)
        .await
        .expect_err("missing token must be rejected");

    assert_eq!(error.code(), tonic::Code::Unauthenticated);
}
