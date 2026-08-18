//! The `RaftPeer` gRPC service: turns incoming peer RPCs into `Event`s fed
//! to the actor, and turns its reply back into the RPC response.

use tonic::{Request, Response, Status};

use crate::actor::ActorHandle;
use crate::dto;
use crate::proto::raft_peer_server::RaftPeer;
use crate::proto::{
    AppendEntriesReply, AppendEntriesRequest, RequestVoteReply, RequestVoteRequest,
};
use kurogane_raft::Message;

pub struct RaftPeerService {
    actor: ActorHandle,
}

impl RaftPeerService {
    pub fn new(actor: ActorHandle) -> Self {
        Self { actor }
    }
}

#[tonic::async_trait]
impl RaftPeer for RaftPeerService {
    async fn request_vote(
        &self,
        request: Request<RequestVoteRequest>,
    ) -> Result<Response<RequestVoteReply>, Status> {
        let value = dto::request_vote_from_proto(request.into_inner());
        let from = value.candidate_id;

        let reply = self
            .actor
            .peer_request(from, Message::RequestVote(value))
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?
            .ok_or_else(|| Status::invalid_argument("not a recognized cluster member"))?;

        match reply {
            Message::RequestVoteResponse(response) => {
                Ok(Response::new(dto::request_vote_response_to_proto(response)))
            }
            _ => Err(Status::internal(
                "actor returned an unexpected response type",
            )),
        }
    }

    async fn append_entries(
        &self,
        request: Request<AppendEntriesRequest>,
    ) -> Result<Response<AppendEntriesReply>, Status> {
        let value = dto::append_entries_from_proto(request.into_inner());
        let from = value.leader_id;

        let reply = self
            .actor
            .peer_request(from, Message::AppendEntries(value))
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?
            .ok_or_else(|| Status::invalid_argument("not a recognized cluster member"))?;

        match reply {
            Message::AppendEntriesResponse(response) => Ok(Response::new(
                dto::append_entries_response_to_proto(response),
            )),
            _ => Err(Status::internal(
                "actor returned an unexpected response type",
            )),
        }
    }
}
