//! Outbound peer RPC client: implements `PeerTransport` over real gRPC.

use std::collections::BTreeMap;

use kurogane_raft::{Message, NodeId};
use tonic::Request;
use tonic::transport::{Channel, Endpoint};

use crate::actor::{ActorHandle, PeerTransport};
use crate::auth::attach_token;
use crate::dto;
use crate::proto::raft_peer_client::RaftPeerClient;

/// Sends `Effect::Send` messages to peers over real gRPC, spawning a task
/// per call so a slow or unreachable peer never blocks the actor loop —
/// `PeerTransport::send` must not block, and a Tokio task is what makes
/// that hold even though the actual RPC is async.
pub struct GrpcPeerTransport {
    clients: BTreeMap<NodeId, RaftPeerClient<Channel>>,
    actor: ActorHandle,
    token: String,
}

impl GrpcPeerTransport {
    /// `addresses` maps each peer's `NodeId` to its gRPC endpoint (e.g.
    /// `"http://127.0.0.1:50052"`). Connections are lazy — established on
    /// first actual use, not here.
    pub fn new(addresses: BTreeMap<NodeId, String>, token: String, actor: ActorHandle) -> Self {
        let clients = addresses
            .into_iter()
            .map(|(id, address)| {
                let endpoint =
                    Endpoint::from_shared(address).expect("configured peer address is a valid URI");
                (id, RaftPeerClient::new(endpoint.connect_lazy()))
            })
            .collect();
        Self {
            clients,
            actor,
            token,
        }
    }
}

impl PeerTransport for GrpcPeerTransport {
    fn send(&mut self, to: NodeId, message: Message) {
        let Some(client) = self.clients.get(&to).cloned() else {
            return;
        };
        let actor = self.actor.clone();
        let token = self.token.clone();

        tokio::spawn(async move {
            let mut client = client;
            let response = match message {
                Message::RequestVote(request) => {
                    let request =
                        attach_token(Request::new(dto::request_vote_to_proto(request)), &token);
                    client.request_vote(request).await.map(|response| {
                        Message::RequestVoteResponse(dto::request_vote_response_from_proto(
                            response.into_inner(),
                        ))
                    })
                }
                Message::AppendEntries(request) => {
                    let request =
                        attach_token(Request::new(dto::append_entries_to_proto(request)), &token);
                    client.append_entries(request).await.map(|response| {
                        Message::AppendEntriesResponse(dto::append_entries_response_from_proto(
                            response.into_inner(),
                        ))
                    })
                }
                Message::InstallSnapshot(request) => {
                    let request = attach_token(
                        Request::new(dto::install_snapshot_to_proto(request)),
                        &token,
                    );
                    client.install_snapshot(request).await.map(|response| {
                        Message::InstallSnapshotResponse(dto::install_snapshot_response_from_proto(
                            response.into_inner(),
                        ))
                    })
                }
                // kurogane-raft never emits a Send carrying a *response*
                // message -- only requests originate from Node::step.
                Message::RequestVoteResponse(_)
                | Message::AppendEntriesResponse(_)
                | Message::InstallSnapshotResponse(_) => return,
            };

            // A failed call (unreachable peer, timeout, stale token, ...) is
            // dropped -- Raft's own retry logic covers it, same as any
            // other lost message.
            if let Ok(response) = response {
                actor.peer_response(to, response);
            }
        });
    }
}
