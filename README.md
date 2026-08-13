# Kurogane

Kurogane is a from-scratch Rust implementation of Raft and a replicated in-memory key/value state machine. The project is structured as a transport-free deterministic core first, followed by controlled simulation and only then networking, persistence, and client-facing behavior.

## Current status

The repository contains the verified workspace scaffold. Raft transition behavior has not been implemented yet.

- `kurogane-raft` owns protocol types, node state, and synchronous transitions.
- `kurogane-sim` owns deterministic cluster ordering, scheduling, fault injection, traces, and invariant checks.

The current code provides typed node identities, roles, vote messages, events, effects, validated initial node state, and fixed-cluster construction. It intentionally contains no async runtime, sockets, wall-clock timers, random source, storage, protobuf, or gRPC dependency.

## Workspace

```text
crates/
├── kurogane-raft/
└── kurogane-sim/
```

`kurogane-sim` depends on `kurogane-raft`; the core does not depend on the simulator or future runtime layers.

## Development

Requires Rust 1.85 or newer.

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Next milestone

Implement deterministic election safety:

- follower, candidate, and leader transitions
- election ticks and explicit timeout selection
- `RequestVote` and `RequestVoteResponse`
- self-voting and unique quorum accounting
- stale-term rejection and higher-term step-down
- reproducible three-node simulations without sleeps or sockets

Log replication, client commands, persistence, gRPC, snapshots, and membership changes remain out of scope until their earlier invariants are proven.
