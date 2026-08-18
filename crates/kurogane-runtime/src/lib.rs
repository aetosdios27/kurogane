//! Runtime adapter: Tokio task ownership, gRPC transport, and file-based
//! storage on top of `kurogane-raft`/`kurogane-kv`. `tonic`/`prost` types
//! stop at this crate's boundary — they never cross into either of those.

pub mod proto {
    tonic::include_proto!("kurogane");
}

pub mod dto;
