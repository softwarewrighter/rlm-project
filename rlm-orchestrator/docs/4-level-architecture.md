# RLM 4-Level Capability Architecture

## Overview

The RLM (Recursive Language Model) system supports 4 capability levels with increasing power and risk. Each level has its own configuration, controls, and extensibility.

## The Four Capability Levels

| Level | Name | Default | Risk | Use Case |
|-------|------|---------|------|----------|
| 1 | **DSL** | ON | Very Low | Filter/search/slice textual data with JSON ops |
| 2 | **WASM MapReduce** | ON | Low | Sandboxed computation with LLM-generated comparators |
| 3 | **Rust CLI** | OFF | Medium | Full Rust stdlib, databases, networks, no sandbox |
| 4 | **LLM Delegation** | OFF | Variable | Chunk data to specialized LLMs for fuzzy analysis |

## Feature Comparison Table

| Feature | Level 1: DSL | Level 2: WASM | Level 3: CLI | Level 4: LLM |
|---------|--------------|---------------|--------------|--------------|
| **Commands** | slice, lines, regex, find, count, split, len, set, get, print, final | wasm, rust_wasm, rust_wasm_intent, rust_wasm_mapreduce | rust_cli_intent | llm_query, llm_delegate_chunks |
| **Sandboxing** | N/A (read-only) | Fuel + Memory limits | Process isolation only | Provider trust |
| **I/O Access** | None | None | Configurable | External API calls |
| **Determinism** | 100% | 100% | 100% | Non-deterministic |
| **Max Data** | Unlimited | ~64MB per chunk | Unlimited | Chunk-based |
| **Speed** | Fastest | Fast (compile overhead) | Fast (compile overhead) | Slow (LLM latency) |
| **Use Cases** | Text extraction, filtering | Frequency counting, statistics | Complex algorithms, binary data | Summarization, entity extraction |
| **Pros** | Safe, fast, predictable | Sandboxed code execution | Full capability | Semantic understanding |
| **Cons** | Limited operations | WASM restrictions | Security risk | Cost, latency, non-deterministic |

## Level 1: DSL (Domain Specific Language)

### Description
Safe, read-only text operations for extraction and filtering. Always enabled by default.

### Commands
- `slice` - Extract character range
- `lines` - Extract line range
- `regex` - Pattern matching
- `find` - Text search
- `count` - Count lines/chars/words/matches
- `split` - Split text by delimiter
- `len` - Get length
- `set` / `get` - Variable management
- `print` - Output value
- `final` / `final_var` - Return result

### Configuration
```toml
[dsl]
enabled = true
max_regex_matches = 10000
max_slice_size = 1048576
max_variables = 100
max_variable_size = 1048576
```

### Security
- No I/O operations
- Read-only context access
- Memory limits on regex matches and variables

## Level 2: WASM MapReduce

### Description
Sandboxed code execution using WebAssembly with fuel and memory limits. LLM generates Rust code which is compiled to WASM and executed in a sandbox.

### Commands
- `wasm` - Execute pre-compiled WASM module
- `wasm_wat` - Compile and execute WAT code
- `rust_wasm` - Compile and execute Rust code
- `wasm_template` - Template-based WASM with hooks
- `rust_wasm_intent` - LLM generates code from natural language
- `rust_wasm_reduce_intent` - Streaming reduce for large datasets
- `rust_wasm_mapreduce` - Map-reduce pattern (most reliable)

### Configuration
```toml
[wasm]
enabled = true
rust_wasm_enabled = true
fuel_limit = 1000000        # Max WASM instructions
memory_limit = 67108864     # 64MB max memory
cache_size = 100
codegen_provider = "litellm"
codegen_url = "http://localhost:4000"
codegen_model = "deepseek/deepseek-coder"
```

### Security
- Fuel limits prevent infinite loops
- Memory limits prevent exhaustion
- Source code validation blocks dangerous operations
- No filesystem, network, or process access

## Level 3: Rust CLI

### Description
Full Rust capability with native binary compilation. Process isolation only - no WASM sandbox. Disabled by default.

### Commands
- `rust_cli_intent` - LLM generates Rust code, compiled to native binary

### Configuration
```toml
[cli]
enabled = false
sandbox_mode = "none"       # "none", "docker", "seccomp" (future)
timeout_secs = 30
max_output_size = 10485760

[cli.validation]
allow_filesystem_read = false
allow_filesystem_write = false
allow_network = false
allow_process_spawn = false
allow_unsafe = false
```

### Security
- Source code validation (same as WASM)
- Process isolation
- Timeout limits
- Future: Docker/LXC/seccomp sandboxing

### Advantages over WASM
- Full Rust standard library (HashMap, regex, all string ops)
- No memory limits
- Faster execution for large datasets
- No TwoWaySearcher/memcmp crashes

## Level 4: LLM Delegation

### Description
Delegate chunks of text to specialized LLMs for fuzzy/semantic analysis. Non-deterministic. Disabled by default.

### Commands
- `llm_query` - Call sub-LLM with prompt
- `llm_delegate_chunks` - Chunk content and process with LLM (future)

### Configuration
```toml
[llm_delegation]
enabled = false
chunk_size = 4096
overlap = 256
max_chunks = 100
privacy_mode = "local"      # "local", "cloud", "hybrid"
max_concurrent = 5
rate_limit_per_minute = 60

[llm_delegation.provider]
type = "litellm"
url = "http://localhost:4000"
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"
```

### Privacy Modes
- **local**: Use local LLM only (privacy-preserving)
- **cloud**: Use cloud LLM (faster, may expose data)
- **hybrid**: Use local for sensitive content, cloud for general

## CLI Arguments

### Level Control
```bash
# Default: DSL + WASM only (safe)
rlm document.txt "Count errors"

# Enable specific levels
rlm large.log "Rank IPs" --enable-cli
rlm article.txt "Summarize" --enable-llm-delegation

# Enable all levels
rlm data.txt "Complex query" --enable-all

# Explicit level selection
rlm data.txt "Query" --levels dsl,wasm,cli

# Disable levels
rlm data.txt "Query" --disable-wasm

# Set priority
rlm data.txt "Query" --priority cli,wasm,dsl
```

### Backward Compatibility
- `--no-wasm` → `--disable-wasm`
- `--no-rust-wasm` → Sets `wasm.rust_wasm_enabled = false`

## Best Practices

1. **Use the lowest level that can accomplish the task**
   - Simple extraction → Level 1 (DSL)
   - Frequency counting → Level 2 (WASM) or Level 3 (CLI)
   - Semantic analysis → Level 4 (LLM)

2. **Enable CLI (Level 3) only when needed**
   - Large datasets (>10MB)
   - Complex algorithms requiring full stdlib
   - When WASM has issues (TwoWaySearcher crashes)

3. **LLM Delegation (Level 4) considerations**
   - Use local mode for sensitive data
   - Set appropriate rate limits
   - Consider cost vs accuracy tradeoffs

## System Prompt Adaptation

The system prompt automatically adapts based on enabled levels:

```
## Capability Levels
- Level 1 (DSL): Text extraction, filtering [ENABLED]
- Level 2 (WASM): Sandboxed computation [ENABLED]
- Level 3 (CLI): Native binary execution [DISABLED]
- Level 4 (LLM): Semantic analysis delegation [DISABLED]

Use the LOWEST level that can accomplish the task.
```

When CLI is enabled, the LLM is instructed to prefer `rust_cli_intent` for analysis tasks. When only WASM is available, it uses `rust_wasm_mapreduce`.
