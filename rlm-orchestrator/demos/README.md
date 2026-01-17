# RLM Demos

This directory contains demos for RLM (Recursive Language Model), organized by **capability level**. Each demo is available both as a **command-line script** and in the **Web UI visualizer**.

## Capability Levels

RLM supports 4 capability levels, each with different trade-offs:

| Level | Name | Description | Sandbox | Use Case |
|-------|------|-------------|---------|----------|
| **L1** | DSL | Text operations (slice, find, regex, count) | Safe | Simple pattern matching, line extraction |
| **L2** | WASM | Sandboxed Rust computation | Safe | Aggregation, frequency counting, statistics |
| **L3** | CLI | Native Rust binary (future) | Unsafe | Complex analysis requiring full stdlib |
| **L4** | Recursive LLM | Multi-hop LLM reasoning (future) | Safe | Semantic analysis of scattered information |

## Problem Types (from RLM Paper)

- **Simple NIAH**: "Needle in a haystack" - find specific patterns in large text
- **OOLONG**: Aggregation and statistical operations over data
- **BrowseComp-Plus**: Complex extraction and ranking tasks
- **S-NIAH**: Scattered needle - information spread across document

---

## Demo Correspondence: CLI Scripts ↔ Web UI

### Level 1: DSL (Text Operations)

| CLI Script | Web UI Dropdown | Problem Type | Description |
|------------|-----------------|--------------|-------------|
| `l1/error-count.sh` | "Count ERROR lines" | Simple NIAH | Count lines matching a pattern using DSL find/count |

### Level 2: WASM (Sandboxed Computation)

| CLI Script | Web UI Dropdown | Problem Type | Description |
|------------|-----------------|--------------|-------------|
| `l2/unique-ips.sh` | "Count unique IP addresses" | OOLONG | Count unique items using rust_wasm_mapreduce (combiner=unique) |
| `l2/error-ranking.sh` | "Rank errors by frequency" | OOLONG | Frequency ranking using rust_wasm_mapreduce (combiner=count) |
| `l2/percentiles.sh` | "Response time percentiles" | OOLONG | Statistical computation using rust_wasm_intent (needs sorting) |
| - | "HTTP status code frequency" | OOLONG | Similar to error-ranking but for HTTP status codes |
| - | "Large Logs: Error Ranking" | BrowseComp-Plus | Same as error-ranking but with 5000 lines |
| - | "Large Logs: Unique IPs" | BrowseComp-Plus | Same as unique-ips but with 5000 lines |

### Level 3: CLI (Native Binary) - FUTURE

| CLI Script | Web UI Dropdown | Problem Type | Description |
|------------|-----------------|--------------|-------------|
| `future/l3-cli/*` | "Word frequency analysis" | OOLONG | Complex text processing with full Rust stdlib |

### Level 4: Recursive LLM - FUTURE

| CLI Script | Web UI Dropdown | Problem Type | Description |
|------------|-----------------|--------------|-------------|
| `l4/family-tree.sh` | "War and Peace Family Tree" | S-NIAH | Multi-hop reasoning over 3.2MB text |

---

## Running Demos

### Prerequisites

1. **Build the CLI**:
   ```bash
   cargo build --release
   ```

2. **Set up API keys** in `~/.env`:
   ```bash
   DEEPSEEK_API_KEY=your_key_here
   ```

3. **Start LiteLLM gateway**:
   ```bash
   litellm --config litellm_config.yaml --port 4000
   ```

4. **Start RLM server** (for Web UI and demos that fetch sample data):
   ```bash
   ./scripts/run-server-litellm.sh
   ```

### CLI Scripts

```bash
# Level 1: DSL demo
./demos/l1/error-count.sh

# Level 2: WASM demos (today's focus)
./demos/l2/unique-ips.sh
./demos/l2/error-ranking.sh
./demos/l2/percentiles.sh

# Level 4: Recursive LLM demo (future)
./demos/l4/family-tree.sh
```

### Web UI Visualizer

1. Open http://localhost:8080/visualize
2. Select a demo from the "Load Example" dropdown
3. Click "Run RLM Query"
4. Watch real-time progress and WASM execution

---

## WASM Operations (Level 2)

Level 2 demos use two WASM operations:

### rust_wasm_mapreduce
For per-line extraction + aggregation:
- `combiner="unique"` - count unique items (HashSet)
- `combiner="count"` - frequency counting (HashMap)
- `combiner="sum"` - numeric totals

### rust_wasm_intent
For operations needing ALL data at once:
- Sorting, percentiles, median
- Complex statistics
- Any operation requiring sorted/complete data

---

## Directory Structure

```
demos/
├── README.md           # This file
├── common.sh           # Shared setup (API keys, paths, run_demo function)
├── l1/                 # Level 1: DSL demos
│   └── error-count.sh
├── l2/                 # Level 2: WASM demos
│   ├── unique-ips.sh
│   ├── error-ranking.sh
│   └── percentiles.sh
├── l4/                 # Level 4: Recursive LLM demos
│   └── family-tree.sh
└── future/             # Work in progress
    └── l3-cli/         # Level 3: CLI demos (not yet ready)
```

---

## Expected Performance

All times are estimates using DeepSeek via LiteLLM gateway:

| Demo | Expected Time | Iterations | Notes |
|------|---------------|------------|-------|
| L1: error-count | 15-30s | 1-2 | Simple DSL operations |
| L2: unique-ips | 15-25s | 1-2 | WASM mapreduce unique |
| L2: error-ranking | 15-25s | 1-2 | WASM mapreduce count |
| L2: percentiles | 15-30s | 1-2 | WASM intent (sorting) |
| L4: family-tree | 60-90s | 3-5 | Multi-hop LLM reasoning |

---

## Troubleshooting

### "LiteLLM not responding"
Ensure LiteLLM gateway is running on port 4000:
```bash
litellm --config litellm_config.yaml --port 4000
```

### "Sample data not found"
Start the RLM server for demos that fetch sample data:
```bash
./scripts/run-server-litellm.sh
```

### "WASM compilation failed"
Check that the LLM is generating valid Rust code. The sub-LLM (gemma2:9b via Ollama) handles code generation for WASM operations.
