//! File-based persistence for `HardState`/log, backing `Node::recover`.

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use kurogane_raft::{Effect, HardState, LogEntry, NodeId, SnapshotMetadata};
use prost::Message;

use crate::dto::{log_entry_from_proto, log_entry_to_proto};
use crate::proto;

/// One node's durable storage: a single file holding hard state, log, and
/// snapshot, rewritten and fsynced on every `Persist*` effect. No WAL or
/// checksums — a synchronous write+flush is what the declared crash model
/// actually requires, not more than that.
pub struct Storage {
    path: PathBuf,
    hard_state: HardState,
    log: Vec<LogEntry>,
    snapshot: SnapshotMetadata,
    snapshot_data: Vec<u8>,
}

impl Storage {
    /// Opens (or initializes) durable storage at `path`. A missing file
    /// means a fresh node: default hard state, empty log, no snapshot.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        let path = path.into();
        let (hard_state, log, snapshot, snapshot_data) = match fs::read(&path) {
            Ok(bytes) => {
                let record = proto::StorageState::decode(bytes.as_slice())
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                (
                    HardState {
                        current_term: record.current_term,
                        voted_for: record.voted_for.map(NodeId),
                    },
                    record.log.into_iter().map(log_entry_from_proto).collect(),
                    SnapshotMetadata {
                        last_included_index: record.snapshot_last_included_index,
                        last_included_term: record.snapshot_last_included_term,
                    },
                    record.snapshot_data,
                )
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => (
                HardState::default(),
                Vec::new(),
                SnapshotMetadata::default(),
                Vec::new(),
            ),
            Err(error) => return Err(error),
        };

        Ok(Self {
            path,
            hard_state,
            log,
            snapshot,
            snapshot_data,
        })
    }

    pub fn hard_state(&self) -> HardState {
        self.hard_state
    }

    pub fn log(&self) -> &[LogEntry] {
        &self.log
    }

    pub fn snapshot(&self) -> SnapshotMetadata {
        self.snapshot
    }

    pub fn snapshot_data(&self) -> &[u8] {
        &self.snapshot_data
    }

    /// Records one effect as durable, synchronously. `Send` is not
    /// persistence and is ignored, matching `kurogane-sim`'s in-memory
    /// `DurableState`. The caller must not act on any dependent `Send`
    /// until this returns `Ok`.
    pub fn apply(&mut self, effect: &Effect) -> io::Result<()> {
        match effect {
            Effect::PersistHardState { term, voted_for } => {
                self.hard_state = HardState {
                    current_term: *term,
                    voted_for: *voted_for,
                };
                self.flush()
            }
            Effect::PersistLog {
                from_index,
                entries,
            } => {
                // `from_index` is an absolute log index; `log[0]` holds
                // whatever comes right after the current snapshot boundary,
                // not necessarily absolute index 1.
                self.log
                    .truncate((*from_index - self.snapshot.last_included_index - 1) as usize);
                self.log.extend(entries.iter().cloned());
                self.flush()
            }
            Effect::PersistSnapshot {
                last_included_index,
                last_included_term,
                data,
            } => {
                self.snapshot = SnapshotMetadata {
                    last_included_index: *last_included_index,
                    last_included_term: *last_included_term,
                };
                self.snapshot_data = data.clone();
                self.flush()
            }
            Effect::Send { .. } => Ok(()),
        }
    }

    fn flush(&self) -> io::Result<()> {
        let record = proto::StorageState {
            current_term: self.hard_state.current_term,
            voted_for: self.hard_state.voted_for.map(|id| id.0),
            log: self.log.iter().cloned().map(log_entry_to_proto).collect(),
            snapshot_last_included_index: self.snapshot.last_included_index,
            snapshot_last_included_term: self.snapshot.last_included_term,
            snapshot_data: self.snapshot_data.clone(),
        };

        let mut file = fs::File::create(&self.path)?;
        file.write_all(&record.encode_to_vec())?;
        file.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use kurogane_raft::{HardState, LogEntry, NodeId};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn opening_a_missing_file_yields_fresh_state() {
        let dir = tempdir().expect("temp dir");
        let storage = Storage::open(dir.path().join("state")).expect("open storage");

        assert_eq!(storage.hard_state(), HardState::default());
        assert!(storage.log().is_empty());
    }

    #[test]
    fn persist_hard_state_survives_a_reopen() {
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("state");

        let mut storage = Storage::open(&path).expect("open storage");
        storage
            .apply(&Effect::PersistHardState {
                term: 3,
                voted_for: Some(NodeId(2)),
            })
            .expect("persist hard state");

        let reopened = Storage::open(&path).expect("reopen storage");
        assert_eq!(
            reopened.hard_state(),
            HardState {
                current_term: 3,
                voted_for: Some(NodeId(2)),
            }
        );
    }

    #[test]
    fn persist_log_splices_and_survives_a_reopen() {
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("state");

        let mut storage = Storage::open(&path).expect("open storage");
        storage
            .apply(&Effect::PersistLog {
                from_index: 1,
                entries: vec![
                    LogEntry {
                        term: 1,
                        command: vec![1],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![2],
                    },
                ],
            })
            .expect("persist log");

        // A conflict-truncate at index 2, same as on_append_entries applies
        // in-memory.
        let replacement = LogEntry {
            term: 2,
            command: vec![9],
        };
        storage
            .apply(&Effect::PersistLog {
                from_index: 2,
                entries: vec![replacement.clone()],
            })
            .expect("persist log");

        let reopened = Storage::open(&path).expect("reopen storage");
        assert_eq!(reopened.log().len(), 2);
        assert_eq!(reopened.log()[1], replacement);
    }

    #[test]
    fn persist_snapshot_then_persist_log_uses_the_new_boundary_for_absolute_indexing() {
        let dir = tempdir().expect("temp dir");
        let path = dir.path().join("state");

        let mut storage = Storage::open(&path).expect("open storage");
        storage
            .apply(&Effect::PersistLog {
                from_index: 1,
                entries: vec![
                    LogEntry {
                        term: 1,
                        command: vec![1],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![2],
                    },
                    LogEntry {
                        term: 1,
                        command: vec![3],
                    },
                ],
            })
            .expect("persist log");

        // Compacting through index 3, mirroring exactly what Node::compact
        // emits: PersistSnapshot moves the boundary, then a PersistLog
        // pins down what's retained above it -- here, nothing.
        storage
            .apply(&Effect::PersistSnapshot {
                last_included_index: 3,
                last_included_term: 1,
                data: vec![9, 9],
            })
            .expect("persist snapshot");
        storage
            .apply(&Effect::PersistLog {
                from_index: 4,
                entries: Vec::new(),
            })
            .expect("persist log");

        // A later entry lands at absolute index 4 -- the first index above
        // the new boundary, not vec position 4 counted from the old start.
        storage
            .apply(&Effect::PersistLog {
                from_index: 4,
                entries: vec![LogEntry {
                    term: 1,
                    command: vec![4],
                }],
            })
            .expect("persist log");

        let reopened = Storage::open(&path).expect("reopen storage");
        assert_eq!(
            reopened.snapshot(),
            SnapshotMetadata {
                last_included_index: 3,
                last_included_term: 1,
            }
        );
        assert_eq!(reopened.snapshot_data(), &[9, 9]);
        assert_eq!(
            reopened.log(),
            &[LogEntry {
                term: 1,
                command: vec![4]
            }]
        );
    }

    #[test]
    fn ignores_send_effects() {
        use kurogane_raft::{Message, RequestVoteResponse};

        let dir = tempdir().expect("temp dir");
        let mut storage = Storage::open(dir.path().join("state")).expect("open storage");

        storage
            .apply(&Effect::Send {
                to: NodeId(2),
                message: Message::RequestVoteResponse(RequestVoteResponse {
                    term: 1,
                    granted: true,
                }),
            })
            .expect("apply is a no-op for Send");

        assert_eq!(storage.hard_state(), HardState::default());
        assert!(storage.log().is_empty());
        assert!(!dir.path().join("state").exists());
    }
}
