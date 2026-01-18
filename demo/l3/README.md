# Level 3: CLI Demos (Native Binary Execution)

These demos showcase the L3 CLI capability for complex analysis tasks that benefit from native Rust execution instead of WASM sandboxing.

## Why L3 CLI?

L3 CLI uses `rust_cli_intent` which compiles and runs native Rust binaries. This is preferred for:

- **Large datasets (1000+ lines)** - WASM may timeout due to fuel limits
- **Complex aggregations** - HashMap/HashSet-based frequency counting
- **Statistical operations** - Sorting, percentiles, statistics
- **When WASM fails** - TwoWaySearcher panics, memory limits, etc.

## L1+L3 Hybrid Approach

These demos use a hybrid approach:
1. **L1 DSL** pre-filters data (find, regex, lines)
2. **L3 CLI** processes the filtered data with native code

This reduces input size and simplifies the generated Rust code.

## Demo Scripts

| Script | Problem Type | Dataset | Description |
|--------|--------------|---------|-------------|
| `error-ranking.sh` | BrowseComp-Plus | 5000 log lines | Rank error types by frequency |
| `unique-ips.sh` | BrowseComp-Plus | 5000 log lines | Count unique IPs, show top 10 |
| `percentiles.sh` | OOLONG | 2000 entries | Calculate p50, p95, p99 response times |

## Web UI Correspondence

Each demo script has a corresponding example in the Web Visualizer:

| CLI Demo | Web UI Example | URL |
|----------|----------------|-----|
| `error-ranking.sh` | CLI: Error Ranking (5000 lines) | `/visualize` |
| `unique-ips.sh` | CLI: Unique IPs (5000 lines) | `/visualize` |
| `percentiles.sh` | CLI: Response Percentiles | `/visualize` |

## Running the Demos

### Prerequisites

1. Build the RLM CLI binary:
   ```bash
   cd rlm-orchestrator
   cargo build --release
   ```

2. Start the RLM server with CLI enabled:
   ```bash
   cargo run --bin rlm-server -- --enable-cli
   ```

### Running Demos

```bash
# From project root
./demo/l3/error-ranking.sh
./demo/l3/unique-ips.sh
./demo/l3/percentiles.sh
```

## Expected Performance

L3 CLI should complete these tasks in 2-5 iterations, compared to WASM which may take 10-20 iterations or timeout on the same 5000-line datasets.

| Task | WASM Iterations | CLI Iterations |
|------|-----------------|----------------|
| Error Ranking (5000 lines) | 15-20 (may timeout) | 2-5 |
| Unique IPs (5000 lines) | 15-20 (may timeout) | 2-5 |
| Percentiles | 10-15 | 2-4 |

## Security Note (PoC)

**L3 CLI Warning**: Native binary execution without sandbox.

The L3 CLI mode executes compiled Rust code as a native binary. While the code generator includes validation that blocks filesystem, network, and process operations, this is **not fully sandboxed**.

For PoC/demo purposes only - production use requires container isolation (e.g., Docker, Firecracker).

Blocked operations:
- File system access (`std::fs`, `std::io::BufReader`, etc.)
- Network access (`std::net`, `reqwest`, etc.)
- Process spawning (`std::process::Command`, etc.)
- Environment access (`std::env`, etc.)

The validation is implemented in `rlm-orchestrator/src/levels/cli.rs`.
