//! Replicated in-memory key/value state machine on top of `kurogane-raft`.

use std::error::Error;
use std::fmt;

/// A client operation against the replicated key/value state machine.
/// `Get` is a command, not a side-channel read: routing it through the log
/// is what makes it linearizable without a heartbeat-lease mechanism.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Command {
    Set { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Get { key: Vec<u8> },
}

/// Invalid or truncated command bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    Truncated,
    UnknownTag(u8),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => {
                formatter.write_str("command bytes ended before a complete value was read")
            }
            Self::UnknownTag(tag) => write!(formatter, "unknown command tag {tag}"),
        }
    }
}

impl Error for DecodeError {}

const SET_TAG: u8 = 0;
const DELETE_TAG: u8 = 1;
const GET_TAG: u8 = 2;

impl Command {
    /// Encodes this command as a self-describing byte string: a one-byte
    /// tag followed by `u32`-length-prefixed byte strings. Hand-rolled
    /// rather than pulling in `serde`, matching the project's existing
    /// zero-dependency style; a real wire format arrives with milestone six.
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        match self {
            Self::Set { key, value } => {
                bytes.push(SET_TAG);
                encode_bytes(&mut bytes, key);
                encode_bytes(&mut bytes, value);
            }
            Self::Delete { key } => {
                bytes.push(DELETE_TAG);
                encode_bytes(&mut bytes, key);
            }
            Self::Get { key } => {
                bytes.push(GET_TAG);
                encode_bytes(&mut bytes, key);
            }
        }
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut reader = ByteReader::new(bytes);
        match reader.read_u8()? {
            SET_TAG => {
                let key = reader.read_bytes()?;
                let value = reader.read_bytes()?;
                Ok(Self::Set { key, value })
            }
            DELETE_TAG => Ok(Self::Delete {
                key: reader.read_bytes()?,
            }),
            GET_TAG => Ok(Self::Get {
                key: reader.read_bytes()?,
            }),
            other => Err(DecodeError::UnknownTag(other)),
        }
    }
}

fn encode_bytes(buf: &mut Vec<u8>, value: &[u8]) {
    buf.extend_from_slice(&(value.len() as u32).to_be_bytes());
    buf.extend_from_slice(value);
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, DecodeError> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or(DecodeError::Truncated)?;
        self.position += 1;
        Ok(byte)
    }

    fn read_bytes(&mut self) -> Result<Vec<u8>, DecodeError> {
        let length_bytes = self
            .bytes
            .get(self.position..self.position + 4)
            .ok_or(DecodeError::Truncated)?;
        let length = u32::from_be_bytes(length_bytes.try_into().expect("checked length")) as usize;
        self.position += 4;

        let value = self
            .bytes
            .get(self.position..self.position + length)
            .ok_or(DecodeError::Truncated)?;
        self.position += length;
        Ok(value.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, DecodeError};

    #[test]
    fn round_trips_set() {
        let command = Command::Set {
            key: vec![1, 2, 3],
            value: vec![9, 9],
        };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_delete() {
        let command = Command::Delete { key: vec![7] };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_get() {
        let command = Command::Get { key: vec![4, 5] };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn round_trips_empty_keys_and_values() {
        let command = Command::Set {
            key: Vec::new(),
            value: Vec::new(),
        };

        assert_eq!(Command::decode(&command.encode()), Ok(command));
    }

    #[test]
    fn rejects_an_unknown_tag() {
        assert_eq!(Command::decode(&[99]), Err(DecodeError::UnknownTag(99)));
    }

    #[test]
    fn rejects_empty_bytes() {
        assert_eq!(Command::decode(&[]), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_a_truncated_length_prefix() {
        assert_eq!(Command::decode(&[0, 0, 0]), Err(DecodeError::Truncated));
    }

    #[test]
    fn rejects_a_truncated_value() {
        // Tag 0 (Set), key length 5, but only 2 key bytes follow.
        assert_eq!(
            Command::decode(&[0, 0, 0, 0, 5, 1, 2]),
            Err(DecodeError::Truncated)
        );
    }
}
