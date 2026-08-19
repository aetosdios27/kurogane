//! Conversions between the generated protobuf types and `kurogane-raft`'s /
//! `kurogane-kv`'s own types. `tonic`/`prost` types stop here — they never
//! cross into either of those crates.

use std::error::Error;
use std::fmt;

use kurogane_kv::Command;
use kurogane_raft::{
    AppendEntries, AppendEntriesResponse, InstallSnapshot, InstallSnapshotResponse, LogEntry,
    NodeId, RequestVote, RequestVoteResponse,
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

pub fn log_entry_from_proto(proto: proto::LogEntry) -> LogEntry {
    LogEntry {
        term: proto.term,
        command: proto.command,
    }
}

pub fn log_entry_to_proto(value: LogEntry) -> proto::LogEntry {
    proto::LogEntry {
        term: value.term,
        command: value.command,
    }
}

pub fn append_entries_from_proto(proto: proto::AppendEntriesRequest) -> AppendEntries {
    AppendEntries {
        term: proto.term,
        leader_id: NodeId(proto.leader_id),
        prev_log_index: proto.prev_log_index,
        prev_log_term: proto.prev_log_term,
        entries: proto
            .entries
            .into_iter()
            .map(log_entry_from_proto)
            .collect(),
        leader_commit: proto.leader_commit,
    }
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
    }
}

pub fn install_snapshot_to_proto(value: InstallSnapshot) -> proto::InstallSnapshotRequest {
    proto::InstallSnapshotRequest {
        term: value.term,
        leader_id: value.leader_id.0,
        last_included_index: value.last_included_index,
        last_included_term: value.last_included_term,
        data: value.data,
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
                    command: vec![1, 2, 3],
                },
                LogEntry {
                    term: 4,
                    command: Vec::new(),
                },
            ],
            leader_commit: 1,
        };

        assert_eq!(
            append_entries_from_proto(append_entries_to_proto(value.clone())),
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
}
