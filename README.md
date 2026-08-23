# Kurogane

Kurogane is a from-scratch Rust implementation of Raft and a replicated in-memory key/value state machine. It was built transport-free deterministic core first, then controlled simulation, and only later networking, persistence, and client-facing behavior — not an imported Raft library and not aimed at being production-ready today.

## Status: feature-frozen

All nine roadmap milestones are complete and the project has been reconciled end-to-end against the Raft paper (Figure 2/3, §5.4, §5.6, §6, §7, §8). One genuine correctness gap found during that audit — a local-membership check wrongly gating `AppendEntries`/`InstallSnapshot`, which could permanently strand a follower partitioned across a membership change — was fixed and regression-tested. Everything else is either compliant or an intentional, documented deviation.

Implemented:

- Elections, heartbeats, and `RequestVote` log-freshness voting
- Log replication with conflict detection/truncation and leader-tracked `next_index`/`match_index`
- A replicated key/value state machine (`Set`/`Delete`/`Get`, `Get` routed through the log for linearizability)
- Hard-state/log persistence with crash recovery
- A real Tokio/gRPC runtime (`kurogane-runtime`): real sockets, real disk I/O, authenticated peer identity, leader-redirecting clients
- Snapshots and log compaction, including the `InstallSnapshot` RPC for catching up a lagging peer
- Cluster membership changes via joint consensus (dual-majority quorum, learners, config-triggered leader step-down, a removed-server disruption guard)
- Client-visible consistency verification: `Propose` blocks until its index applies and returns the real value; a seeded, fault-injecting harness (`kurogane-verify`) drives real replicas through partition/crash/restart scenarios checked against a hand-rolled linearizability checker

"Feature-frozen" means frozen at this documented scope, not "nothing left to build." Deliberately still open, by documented choice, not oversight:

- Snapshot chunking / resumable transfer
- Client-retry deduplication (the paper's §8 serial-number mechanism — safe today only because nothing in this project retries a `Propose` after an `Indeterminate` outcome)
- Real-process network partition/delay fault injection (real-process fault injection today is kill/restart only)
- A real-process history generator (the linearizability checker is currently exercised only via the deterministic simulator)

## Workspace

```text
crates/
├── kurogane-raft/     # protocol types, node state, synchronous transitions (std only)
├── kurogane-sim/      # deterministic cluster scheduling, fault injection, traces
├── kurogane-kv/       # replicated key/value state machine
├── kurogane-runtime/  # Tokio/gRPC runtime, file-based storage, kurogane-node binary
└── kurogane-verify/   # seeded fault-injecting harness + linearizability checker
```

`kurogane-sim`, `kurogane-kv`, and `kurogane-runtime` depend one-directionally on `kurogane-raft`, which depends on nothing but the standard library — no async runtime, sockets, wall-clock timers, random source, or protobuf. `kurogane-verify` depends on `kurogane-sim` and `kurogane-kv`. `unsafe_code = "forbid"` is set workspace-wide.

## Development

Requires Rust 1.85+ (edition 2024).

```bash
cargo fmt --all -- --check                           # formatting check
cargo test --workspace                                # run all tests
cargo clippy --workspace --all-targets -- -D warnings  # lint, warnings are errors
```

See `CLAUDE.md` for full architectural detail.
