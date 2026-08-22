//! Conversions between the generated protobuf types and `kurogane-raft`'s /
//! `kurogane-kv`'s own types. `tonic`/`prost` types stop here — they never
//! cross into either of those crates.

use std::error::Error;
use std::fmt;

use kurogane_kv::{ApplyResult, Command};
use kurogane_raft::{
    AppendEntries, AppendEntriesResponse, ClusterConfig, InstallSnapshot, InstallSnapshotResponse,
    LogEntry, LogPayload, NodeId, RequestVote, RequestVoteResponse,
};

use crate::proto;

pub fn request_vote_from_proto(proto: proto::RequestVoteRequest) -> RequestVote {
    RequestVote {
        term: proto.term,
        candidate_id: NodeId(proto.candidate_id),
        last_log_index: proto.last_log_index,
        last_log_term: proto.last_log_term,
    }
}

pub fn request_vote_to_proto(value: RequestVote) -> proto::RequestVoteRequest {
    proto::RequestVoteRequest {
        term: value.term,
        candidate_id: value.candidate_id.0,
        last_log_index: value.last_log_index,
        last_log_term: value.last_log_term,
    }
}

pub fn request_vote_response_from_proto(proto: proto::RequestVoteReply) -> RequestVoteResponse {
    RequestVoteResponse {
        term: proto.term,
        granted: proto.granted,
    }
}

pub fn request_vote_response_to_proto(value: RequestVoteResponse) -> proto::RequestVoteReply {
    proto::RequestVoteReply {
        term: value.term,
        granted: value.granted,
    }
}

pub fn configuration_from_proto(proto: proto::Configuration) -> ClusterConfig {
    ClusterConfig {
        voters: proto.voters.into_iter().map(NodeId).collect(),
        old_voters: if proto.old_voters.is_empty() {
            None
        } else {
            Some(proto.old_voters.into_iter().map(NodeId).collect())
        },
    }
}

pub fn configuration_to_proto(value: ClusterConfig) -> proto::Configuration {
    proto::Configuration {
        voters: value.voters.into_iter().map(|id| id.0).collect(),
        old_voters: value
            .old_voters
            .into_iter()
            .flatten()
            .map(|id| id.0)
            .collect(),
    }
}

pub fn log_entry_from_proto(proto: proto::LogEntry) -> Result<LogEntry, MissingLogEntryPayload> {
    let payload = match proto.payload.ok_or(MissingLogEntryPayload)? {
        proto::log_entry::Payload::Command(bytes) => LogPayload::Command(bytes),
        proto::log_entry::Payload::Configuration(configuration) => {
            LogPayload::Configuration(configuration_from_proto(configuration))
        }
    };
    Ok(LogEntry {
        term: proto.term,
        payload,
    })
}

pub fn log_entry_to_proto(value: LogEntry) -> proto::LogEntry {
    let payload = match value.payload {
        LogPayload::Command(bytes) => proto::log_entry::Payload::Command(bytes),
        LogPayload::Configuration(configuration) => {
            proto::log_entry::Payload::Configuration(configuration_to_proto(configuration))
        }
    };
    proto::LogEntry {
        term: value.term,
        payload: Some(payload),
    }
}

pub fn append_entries_from_proto(
    proto: proto::AppendEntriesRequest,
) -> Result<AppendEntries, MissingLogEntryPayload> {
    Ok(AppendEntries {
        term: proto.term,
        leader_id: NodeId(proto.leader_id),
        prev_log_index: proto.prev_log_index,
        prev_log_term: proto.prev_log_term,
        entries: proto
            .entries
            .into_iter()
            .map(log_entry_from_proto)
            .collect::<Result<Vec<_>, _>>()?,
        leader_commit: proto.leader_commit,
    })
}

pub fn append_entries_to_proto(value: AppendEntries) -> proto::AppendEntriesRequest {
    proto::AppendEntriesRequest {
        term: value.term,
        leader_id: value.leader_id.0,
        prev_log_index: value.prev_log_index,
        prev_log_term: value.prev_log_term,
        entries: value.entries.into_iter().map(log_entry_to_proto).collect(),
        leader_commit: value.leader_commit,
    }
}

pub fn append_entries_response_from_proto(
    proto: proto::AppendEntriesReply,
) -> AppendEntriesResponse {
    AppendEntriesResponse {
        term: proto.term,
        success: proto.success,
        match_index: proto.match_index,
    }
}

pub fn append_entries_response_to_proto(value: AppendEntriesResponse) -> proto::AppendEntriesReply {
    proto::AppendEntriesReply {
        term: value.term,
        success: value.success,
        match_index: value.match_index,
    }
}

pub fn install_snapshot_from_proto(proto: proto::InstallSnapshotRequest) -> InstallSnapshot {
    InstallSnapshot {
        term: proto.term,
        leader_id: NodeId(proto.leader_id),
        last_included_index: proto.last_included_index,
        last_included_term: proto.last_included_term,
        data: proto.data,
        // Every real sender (replicate_to) always populates this with its
        // own non-empty snapshot_config -- recover validates a non-empty
        // peers set, and compact/on_install_snapshot both keep
        // snapshot_config in lockstep with it from then on -- so an absent
        // Configuration here would mean a malformed message, not a
        // legitimate "no config" case. unwrap_or_default rather than a
        // fallible Result: a genuinely absent config on this specific
        // field can't happen given the invariant above, unlike
        // log_entry_from_proto's payload oneof, which really can be unset
        // on the wire.
        config: proto
            .config
            .map(configuration_from_proto)
            .unwrap_or_default(),
    }
}

pub fn install_snapshot_to_proto(value: InstallSnapshot) -> proto::InstallSnapshotRequest {
    proto::InstallSnapshotRequest {
        term: value.term,
        leader_id: value.leader_id.0,
        last_included_index: value.last_included_index,
        last_included_term: value.last_included_term,
        data: value.data,
        config: Some(configuration_to_proto(value.config)),
    }
}

pub fn install_snapshot_response_from_proto(
    proto: proto::InstallSnapshotReply,
) -> InstallSnapshotResponse {
    InstallSnapshotResponse {
        term: proto.term,
        last_included_index: proto.last_included_index,
    }
}

pub fn install_snapshot_response_to_proto(
    value: InstallSnapshotResponse,
) -> proto::InstallSnapshotReply {
    proto::InstallSnapshotReply {
        term: value.term,
        last_included_index: value.last_included_index,
    }
}

pub fn propose_config_change_from_proto(proto: proto::ProposeConfigChangeRequest) -> Vec<NodeId> {
    proto.voters.into_iter().map(NodeId).collect()
}

pub fn propose_config_change_to_proto(voters: Vec<NodeId>) -> proto::ProposeConfigChangeRequest {
    proto::ProposeConfigChangeRequest {
        voters: voters.into_iter().map(|id| id.0).collect(),
    }
}

/// `(id, address)` -- this RPC has no `kurogane-raft`/`kurogane-kv` core
/// type to mirror (unlike every other pair in this file): `address` is a
/// purely runtime-level, transport-reachability concept `Node::add_learner`
/// itself has no notion of.
pub fn add_learner_from_proto(proto: proto::AddLearnerRequest) -> (NodeId, String) {
    (NodeId(proto.node_id), proto.address)
}

pub fn add_learner_to_proto(id: NodeId, address: String) -> proto::AddLearnerRequest {
    proto::AddLearnerRequest {
        node_id: id.0,
        address,
    }
}

/// A `Command` message with no `kind` set — malformed input, since every
/// legitimate caller sets exactly one.
#[derive(Debug)]
pub struct MissingCommandKind;

impl fmt::Display for MissingCommandKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Command message is missing its kind")
    }
}

impl Error for MissingCommandKind {}

/// A `LogEntry` message with no `payload` set — malformed input, since
/// every legitimate caller sets either `command` or `configuration`.
#[derive(Debug)]
pub struct MissingLogEntryPayload;

impl fmt::Display for MissingLogEntryPayload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("LogEntry message is missing its payload")
    }
}

impl Error for MissingLogEntryPayload {}

pub fn command_from_proto(proto: proto::Command) -> Result<Command, MissingCommandKind> {
    match proto.kind.ok_or(MissingCommandKind)? {
        proto::command::Kind::Set(set) => Ok(Command::Set {
            key: set.key,
            value: set.value,
        }),
        proto::command::Kind::Delete(delete) => Ok(Command::Delete { key: delete.key }),
        proto::command::Kind::Get(get) => Ok(Command::Get { key: get.key }),
    }
}

pub fn command_to_proto(value: Command) -> proto::Command {
    let kind = match value {
        Command::Set { key, value } => proto::command::Kind::Set(proto::SetCommand { key, value }),
        Command::Delete { key } => proto::command::Kind::Delete(proto::DeleteCommand { key }),
        Command::Get { key } => proto::command::Kind::Get(proto::GetCommand { key }),
    };
    proto::Command { kind: Some(kind) }
}

/// An `ApplyResult` message with no `kind` set — malformed input, since
/// every legitimate caller sets exactly one.
#[derive(Debug)]
pub struct MissingApplyResultKind;

impl fmt::Display for MissingApplyResultKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApplyResult message is missing its kind")
    }
}

impl Error for MissingApplyResultKind {}

pub fn apply_result_from_proto(
    proto: proto::ApplyResult,
) -> Result<ApplyResult, MissingApplyResultKind> {
    match proto.kind.ok_or(MissingApplyResultKind)? {
        proto::apply_result::Kind::Set(set) => Ok(ApplyResult::Set {
            previous: set.previous,
        }),
        proto::apply_result::Kind::Delete(delete) => Ok(ApplyResult::Delete {
            previous: delete.previous,
        }),
        proto::apply_result::Kind::Get(get) => Ok(ApplyResult::Get { value: get.value }),
    }
}

pub fn apply_result_to_proto(value: ApplyResult) -> proto::ApplyResult {
    let kind = match value {
        ApplyResult::Set { previous } => {
            proto::apply_result::Kind::Set(proto::SetResult { previous })
        }
        ApplyResult::Delete { previous } => {
            proto::apply_result::Kind::Delete(proto::DeleteResult { previous })
        }
        ApplyResult::Get { value } => proto::apply_result::Kind::Get(proto::GetResult { value }),
    };
    proto::ApplyResult { kind: Some(kind) }
}

#[cfg(test)]
mod tests {
    use kurogane_kv::Command;
    use kurogane_raft::{
        AppendEntries, AppendEntriesResponse, LogEntry, NodeId, RequestVote, RequestVoteResponse,
    };

    use super::*;

    #[test]
    fn round_trips_request_vote() {
        let value = RequestVote {
            term: 3,
            candidate_id: NodeId(2),
            last_log_index: 7,
            last_log_term: 2,
        };

        assert_eq!(request_vote_from_proto(request_vote_to_proto(value)), value);
    }

    #[test]
    fn round_trips_request_vote_response() {
        let value = RequestVoteResponse {
            term: 3,
            granted: true,
        };

        assert_eq!(
            request_vote_response_from_proto(request_vote_response_to_proto(value)),
            value
        );
    }

    #[test]
    fn round_trips_append_entries_with_entries() {
        let value = AppendEntries {
            term: 4,
            leader_id: NodeId(1),
            prev_log_index: 2,
            prev_log_term: 3,
            entries: vec![
                LogEntry {
                    term: 4,
                    payload: LogPayload::Command(vec![1, 2, 3]),
                },
                LogEntry {
                    term: 4,
                    payload: LogPayload::Command(Vec::new()),
                },
            ],
            leader_commit: 1,
        };

        assert_eq!(
            append_entries_from_proto(append_entries_to_proto(value.clone()))
                .expect("every entry has a payload"),
            value
        );
    }

    #[test]
    fn round_trips_append_entries_response() {
        let value = AppendEntriesResponse {
            term: 4,
            success: true,
            match_index: 9,
        };

        assert_eq!(
            append_entries_response_from_proto(append_entries_response_to_proto(value)),
            value
        );
    }

    #[test]
    fn round_trips_install_snapshot() {
        let value = InstallSnapshot {
            term: 3,
            leader_id: NodeId(2),
            last_included_index: 7,
            last_included_term: 2,
            data: vec![9, 9, 9],
            config: ClusterConfig {
                voters: vec![NodeId(1), NodeId(2)],
                old_voters: Some(vec![NodeId(1), NodeId(3)]),
            },
        };

        assert_eq!(
            install_snapshot_from_proto(install_snapshot_to_proto(value.clone())),
            value
        );
    }

    #[test]
    fn round_trips_install_snapshot_response() {
        let value = InstallSnapshotResponse {
            term: 3,
            last_included_index: 7,
        };

        assert_eq!(
            install_snapshot_response_from_proto(install_snapshot_response_to_proto(value)),
            value
        );
    }

    #[test]
    fn round_trips_set_command() {
        let value = Command::Set {
            key: vec![1],
            value: vec![2, 3],
        };

        assert_eq!(
            command_from_proto(command_to_proto(value.clone())).expect("valid command"),
            value
        );
    }

    #[test]
    fn round_trips_delete_command() {
        let value = Command::Delete { key: vec![9] };

        assert_eq!(
            command_from_proto(command_to_proto(value.clone())).expect("valid command"),
            value
        );
    }

    #[test]
    fn round_trips_get_command() {
        let value = Command::Get { key: vec![4, 5] };

        assert_eq!(
            command_from_proto(command_to_proto(value.clone())).expect("valid command"),
            value
        );
    }

    #[test]
    fn rejects_a_command_with_no_kind_set() {
        let proto = proto::Command { kind: None };

        assert!(command_from_proto(proto).is_err());
    }

    #[test]
    fn round_trips_a_set_apply_result() {
        let value = ApplyResult::Set {
            previous: Some(vec![1, 2]),
        };

        assert_eq!(
            apply_result_from_proto(apply_result_to_proto(value.clone())).expect("valid result"),
            value
        );
    }

    #[test]
    fn round_trips_a_delete_apply_result_with_no_previous_value() {
        let value = ApplyResult::Delete { previous: None };

        assert_eq!(
            apply_result_from_proto(apply_result_to_proto(value.clone())).expect("valid result"),
            value
        );
    }

    #[test]
    fn round_trips_a_get_apply_result() {
        let value = ApplyResult::Get {
            value: Some(vec![9]),
        };

        assert_eq!(
            apply_result_from_proto(apply_result_to_proto(value.clone())).expect("valid result"),
            value
        );
    }

    #[test]
    fn rejects_an_apply_result_with_no_kind_set() {
        let proto = proto::ApplyResult { kind: None };

        assert!(apply_result_from_proto(proto).is_err());
    }

    #[test]
    fn round_trips_a_stable_configuration() {
        let value = ClusterConfig {
            voters: vec![NodeId(1), NodeId(2), NodeId(3)],
            old_voters: None,
        };

        assert_eq!(
            configuration_from_proto(configuration_to_proto(value.clone())),
            value
        );
    }

    #[test]
    fn round_trips_a_joint_configuration() {
        let value = ClusterConfig {
            voters: vec![NodeId(1), NodeId(2), NodeId(4)],
            old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
        };

        assert_eq!(
            configuration_from_proto(configuration_to_proto(value.clone())),
            value
        );
    }

    #[test]
    fn round_trips_a_configuration_log_entry() {
        let value = LogEntry {
            term: 6,
            payload: LogPayload::Configuration(ClusterConfig {
                voters: vec![NodeId(1), NodeId(2)],
                old_voters: Some(vec![NodeId(1), NodeId(2), NodeId(3)]),
            }),
        };

        assert_eq!(
            log_entry_from_proto(log_entry_to_proto(value.clone())).expect("payload is set"),
            value
        );
    }

    #[test]
    fn round_trips_a_propose_config_change_request() {
        let value = vec![NodeId(1), NodeId(2), NodeId(4)];

        assert_eq!(
            propose_config_change_from_proto(propose_config_change_to_proto(value.clone())),
            value
        );
    }

    #[test]
    fn round_trips_an_add_learner_request() {
        let id = NodeId(4);
        let address = "http://127.0.0.1:50054".to_string();

        assert_eq!(
            add_learner_from_proto(add_learner_to_proto(id, address.clone())),
            (id, address)
        );
    }

    #[test]
    fn rejects_a_log_entry_with_no_payload_set() {
        let proto = proto::LogEntry {
            term: 1,
            payload: None,
        };

        assert!(log_entry_from_proto(proto).is_err());
    }
}
