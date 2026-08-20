//! The `RaftPeer` gRPC service: turns incoming peer RPCs into `Event`s fed
//! to the actor, and turns its reply back into the RPC response.

use tonic::{Request, Response, Status};

use crate::actor::{ActorHandle, ProposeOutcome};
use crate::dto;
use crate::proto::raft_client_server::RaftClient;
use crate::proto::raft_peer_server::RaftPeer;
use crate::proto::{
    AppendEntriesReply, AppendEntriesRequest, InstallSnapshotReply, InstallSnapshotRequest,
    NotLeader, ProposeAccepted, ProposeReply, ProposeRequest, RequestVoteReply, RequestVoteRequest,
    propose_reply,
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
        let value = dto::append_entries_from_proto(request.into_inner())
            .map_err(|error| Status::invalid_argument(error.to_string()))?;
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

    async fn install_snapshot(
        &self,
        request: Request<InstallSnapshotRequest>,
    ) -> Result<Response<InstallSnapshotReply>, Status> {
        let value = dto::install_snapshot_from_proto(request.into_inner());
        let from = value.leader_id;

        let reply = self
            .actor
            .peer_request(from, Message::InstallSnapshot(value))
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?
            .ok_or_else(|| Status::invalid_argument("not a recognized cluster member"))?;

        match reply {
            Message::InstallSnapshotResponse(response) => Ok(Response::new(
                dto::install_snapshot_response_to_proto(response),
            )),
            _ => Err(Status::internal(
                "actor returned an unexpected response type",
            )),
        }
    }
}

pub struct RaftClientService {
    actor: ActorHandle,
}

impl RaftClientService {
    pub fn new(actor: ActorHandle) -> Self {
        Self { actor }
    }
}

#[tonic::async_trait]
impl RaftClient for RaftClientService {
    async fn propose(
        &self,
        request: Request<ProposeRequest>,
    ) -> Result<Response<ProposeReply>, Status> {
        let proto = request
            .into_inner()
            .command
            .ok_or_else(|| Status::invalid_argument("ProposeRequest is missing its command"))?;
        let command = dto::command_from_proto(proto)
            .map_err(|error| Status::invalid_argument(error.to_string()))?;

        let outcome = self
            .actor
            .propose(command)
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?;

        let result = match outcome {
            ProposeOutcome::Accepted(index) => {
                propose_reply::Result::Accepted(ProposeAccepted { index })
            }
            ProposeOutcome::NotLeader(hint) => propose_reply::Result::NotLeader(NotLeader {
                leader_id: hint.map(|id| id.0),
            }),
        };
        Ok(Response::new(ProposeReply {
            result: Some(result),
        }))
    }
}
