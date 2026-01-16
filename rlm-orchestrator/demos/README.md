# RLM CLI Demos

This directory contains bash scripts demonstrating RLM's capabilities via the CLI.
All demos use **DeepSeek** via the LiteLLM gateway for both the base LLM and code generation.

**Important:** Do NOT use local Ollama models. Always use LiteLLM gateway.

## Prerequisites

1. **Build the CLI**:
   ```bash
   cargo build --release
   ```

2. **Set up API keys** in `~/.env`:
   ```bash
   DEEPSEEK_API_KEY=your_key_here
   LITELLM_API_KEY=sk-local-test-key-123
   LITELLM_MASTER_KEY=sk-local-test-key-123
   ```

3. **Start the LiteLLM gateway**:
   ```bash
   litellm --config litellm_config.yaml --port 4000
   ```

4. **Start the RLM server** (required for demos that fetch sample data):
   ```bash
   # IMPORTANT: Server needs env vars from ~/.env
   export $(grep -E "^[A-Z]" ~/.env | xargs) && ./target/release/rlm-server config.toml
   ```

   Or use the helper script:
   ```bash
   ./scripts/start-server.sh
   ```

## Demos

### Large Context Demos

These demos require the server running to fetch sample data.

| Demo | Description | Expected Time (DeepSeek) | Iterations |
|------|-------------|--------------------------|------------|
| `war-and-peace-family-tree.sh` | Extracts character relationships from 3.2MB of War and Peace | 60-90 sec | 3-5 |
| `large-logs-error-ranking.sh` | Ranks error types in 5000 log lines using WASM | 30-60 sec | 2-3 |
| `large-logs-unique-ips.sh` | Counts unique IPs in 5000 log lines using WASM | 30-60 sec | 2-3 |

### WASM Demos

These demos generate their own context data.

| Demo | Description | Expected Time (DeepSeek) | Iterations |
|------|-------------|--------------------------|------------|
| `wasm-percentiles.sh` | Calculates p50/p95/p99 response time percentiles | 20-40 sec | 2-3 |

### Basic Demos

These demos use basic RLM commands without WASM.

| Demo | Description | Expected Time (DeepSeek) | Iterations |
|------|-------------|--------------------------|------------|
| `basic-error-count.sh` | Counts ERROR lines in log data | 15-30 sec | 1-2 |

## Running a Demo

```bash
# From the rlm-orchestrator directory
./demos/basic-error-count.sh

# Or from anywhere
/path/to/rlm-orchestrator/demos/war-and-peace-family-tree.sh
```

## Output

Each demo shows:
- Context size (chars, lines)
- Query being executed
- Per-iteration progress (with -v flag):
  - LLM response time and token usage
  - WASM compile/run time (if applicable)
- Final answer
- Total duration

Example output:
```
============================================
Basic Error Count Demo
============================================

Generated: 100 log entries

Query: How many ERROR lines are there?

Running...
----------------------------------------
Starting query (context: 4250 chars, query: 32 chars)
┌─ Iteration 1
│ ⏱  LLM: 2450ms (1234p + 89c tokens)
│ ▶ Extracted 2 command(s)
│   Commands JSON:
│   [{"cmd":"find","args":{"pattern":"ERROR"}}]
│ ◀ Output: 850 chars, 20 lines (2ms)
│   Preview:
│   [2024-01-15 10:05:00] ERROR AuthenticationFailed from 192.168.1.105
│   [2024-01-15 10:10:00] ERROR AuthenticationFailed from 192.168.1.110
└────────────────────────────────────────
┌─ Iteration 2
│ ⏱  LLM: 1890ms (890p + 45c tokens)
│ ✓ Returning FINAL answer
└────────────────────────────────────────
Completed in 2 iteration(s), 4523ms total

FINAL: There are 20 ERROR lines.
```

WASM example (large-logs-error-ranking):
```
┌─ Iteration 1
│ ⏱  LLM: 8500ms (3200p + 450c tokens)
│ 🦀 Generated rust_wasm code (via rust_wasm_intent)
│ 🔧 Compiling Rust to WASM... done (180ms)
│ ⚡ Executing WASM: 12ms
│ ◀ Output: 245 chars, 8 lines (192ms)
│   Preview:
│   AuthenticationFailed: 1250
│   DatabaseError: 890
│   NetworkTimeout: 456
└────────────────────────────────────────
```

## Timing Notes

Times are estimates for DeepSeek API via LiteLLM gateway. Actual times depend on:
- Network latency to DeepSeek API
- API server load
- Context size
- Query complexity

## Configuration

Demos use these defaults (override via environment variables):
```bash
LITELLM_URL=http://localhost:4000
LITELLM_MODEL=deepseek/deepseek-chat
```

To use a different model:
```bash
LITELLM_MODEL=gpt-4o ./demos/basic-error-count.sh
```

## Creating New Demos

1. Source `common.sh` for shared utilities
2. Use `fetch_sample` for server-based context or generate locally
3. Use `run_demo` to execute with timing
4. Add entry to this README

```bash
#!/bin/bash
source "$(dirname "$0")/common.sh"

# Setup and context generation...

QUERY="Your query here"
run_demo "$CONTEXT_FILE" "$QUERY" -v
```
