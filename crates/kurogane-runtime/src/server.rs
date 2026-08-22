//! The `RaftPeer` gRPC service: turns incoming peer RPCs into `Event`s fed
//! to the actor, and turns its reply back into the RPC response.

use std::time::Duration;

use tonic::{Request, Response, Status};

use crate::actor::{ActorHandle, AddLearnerOutcome, ProposeOutcome};
use crate::dto;
use crate::proto::raft_client_server::RaftClient;
use crate::proto::raft_peer_server::RaftPeer;
use crate::proto::{
    AddLearnerAccepted, AddLearnerReply, AddLearnerRequest, AppendEntriesReply,
    AppendEntriesRequest, InstallSnapshotReply, InstallSnapshotRequest, NotLeader, ProposeApplied,
    ProposeConfigChangeAccepted, ProposeConfigChangeReply, ProposeConfigChangeRequest,
    ProposeIndeterminate, ProposeReply, ProposeRequest, RequestVoteReply, RequestVoteRequest,
    add_learner_reply, propose_config_change_reply, propose_reply,
};
use kurogane_raft::Message;

/// How long `Propose` blocks waiting for its assigned index to apply on
/// this node before giving up and reporting `Indeterminate`. Generous
/// relative to `kurogane-node`'s own defaults (a 50ms tick interval, up to
/// a 20-tick/1s election timeout) — this is a liveness bound on one RPC
/// call, not a correctness-sensitive value, so it errs wide rather than
/// tight.
const PROPOSE_APPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// Blocks until `watch` reports `index` (or higher) applied, or the actor
/// task is gone. Returns whether it actually applied — `false` only means
/// the watch channel closed, since the caller wraps this in its own
/// `tokio::time::timeout` for the "still running but too slow" case.
async fn wait_for_applied(mut watch: tokio::sync::watch::Receiver<u64>, index: u64) -> bool {
    loop {
        if *watch.borrow() >= index {
            return true;
        }
        if watch.changed().await.is_err() {
            return false;
        }
    }
}

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

        let index = match outcome {
            ProposeOutcome::Accepted(index) => index,
            ProposeOutcome::NotLeader(hint) => {
                return Ok(Response::new(ProposeReply {
                    result: Some(propose_reply::Result::NotLeader(NotLeader {
                        leader_id: hint.map(|id| id.0),
                    })),
                }));
            }
        };

        let result = match tokio::time::timeout(
            PROPOSE_APPLY_TIMEOUT,
            wait_for_applied(self.actor.applied_watch(), index),
        )
        .await
        {
            Ok(true) => match self.actor.applied_result(index).await.flatten() {
                Some(applied) => propose_reply::Result::Applied(ProposeApplied {
                    index,
                    result: Some(dto::apply_result_to_proto(applied)),
                }),
                // The watch fired but the result is already gone (a very
                // active cluster pruned it via compaction between the two
                // round trips) or the actor task is gone -- either way this
                // caller genuinely can't confirm what happened, same as a
                // plain timeout.
                None => propose_reply::Result::Indeterminate(ProposeIndeterminate {}),
            },
            // Timed out, or the watch channel closed (actor task gone):
            // this index may or may not ever apply -- see ProposeReply's
            // doc comment on Indeterminate in the .proto file.
            _ => propose_reply::Result::Indeterminate(ProposeIndeterminate {}),
        };
        Ok(Response::new(ProposeReply {
            result: Some(result),
        }))
    }

    async fn propose_config_change(
        &self,
        request: Request<ProposeConfigChangeRequest>,
    ) -> Result<Response<ProposeConfigChangeReply>, Status> {
        let new_voters = dto::propose_config_change_from_proto(request.into_inner());

        let outcome = self
            .actor
            .propose_config_change(new_voters)
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?;

        let result = match outcome {
            ProposeOutcome::Accepted(index) => {
                propose_config_change_reply::Result::Accepted(ProposeConfigChangeAccepted { index })
            }
            ProposeOutcome::NotLeader(hint) => {
                propose_config_change_reply::Result::NotLeader(NotLeader {
                    leader_id: hint.map(|id| id.0),
                })
            }
        };
        Ok(Response::new(ProposeConfigChangeReply {
            result: Some(result),
        }))
    }

    async fn add_learner(
        &self,
        request: Request<AddLearnerRequest>,
    ) -> Result<Response<AddLearnerReply>, Status> {
        let (id, address) = dto::add_learner_from_proto(request.into_inner());

        let outcome = self
            .actor
            .add_learner(id, address)
            .await
            .ok_or_else(|| Status::unavailable("actor task is not running"))?;

        let result = match outcome {
            AddLearnerOutcome::Accepted { .. } => {
                add_learner_reply::Result::Accepted(AddLearnerAccepted {})
            }
            AddLearnerOutcome::NotLeader(hint) => add_learner_reply::Result::NotLeader(NotLeader {
                leader_id: hint.map(|id| id.0),
            }),
        };
        Ok(Response::new(AddLearnerReply {
            result: Some(result),
        }))
    }
}
