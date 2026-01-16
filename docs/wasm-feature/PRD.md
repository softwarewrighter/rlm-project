# RLM Dynamic WASM Analysis - Product Requirements Document

## Executive Summary

Enable RLM's root LLM to generate custom Rust analysis functions that are compiled to WebAssembly, loaded into a secure sandbox, and executed against documents too large for LLM context windows. This transforms RLM from a fixed-command interpreter into a programmable analysis engine.

## Problem Statement

### Current Limitations

1. **Fixed Command Set**: RLM currently offers ~10 built-in commands (find, regex, slice, lines, count, etc.). Complex analysis tasks require multiple iterations and creative command chaining.

2. **Semantic Gap**: The LLM must translate sophisticated analysis needs into primitive operations. For example, "extract all email addresses and count by domain" requires:
   - regex to find emails
   - multiple iterations to parse and aggregate
   - manual counting logic

3. **Performance**: Complex multi-step analysis requires many LLM round-trips, increasing latency and token costs.

### Opportunity

WASM execution enables the LLM to write purpose-built analysis code that runs in a single iteration, dramatically improving:
- **Expressiveness**: Any analysis that can be written in Rust
- **Performance**: Single-pass analysis vs. multiple iterations
- **Accuracy**: Direct implementation of complex logic vs. approximation through primitive commands

## Goals

### Primary Goals

1. **Enable Custom Analysis**: LLM can generate Rust code for arbitrary text analysis tasks
2. **Maintain Security**: All generated code runs in a sandboxed WASM environment with strict resource limits
3. **Provide Fast Feedback**: Compilation errors are returned to the LLM for self-correction
4. **Cache Efficiently**: Compiled modules are cached to avoid redundant compilation

### Non-Goals (v1)

- External API calls from WASM modules
- Persistent storage from WASM modules
- Multi-threaded WASM execution
- GPU acceleration
- Network access from WASM

## User Stories

### Story 1: Complex Text Extraction
**As** an RLM user analyzing a large log file,
**I want** to extract all unique IP addresses and count occurrences,
**So that** I can identify the most active clients without manual iteration.

**Current Approach**: 5+ iterations with regex, store, count, llm_query
**With WASM**: Single Rust function that parses, deduplicates, and counts

### Story 2: Custom Data Parsing
**As** an RLM user analyzing a CSV-like document,
**I want** to parse rows and compute statistics on a specific column,
**So that** I can get aggregate metrics directly.

**Current Approach**: Not feasible with current commands
**With WASM**: Rust code with string splitting and arithmetic

### Story 3: Pattern Matching with Context
**As** an RLM user searching legal documents,
**I want** to find all mentions of monetary amounts and extract surrounding sentences,
**So that** I can understand the context of each amount.

**Current Approach**: regex + multiple lines commands + llm_query
**With WASM**: Single function with regex and context extraction

## Functional Requirements

### FR1: Rust Code Generation Command

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { ... }",
  "store": "result"
}
```

The LLM provides Rust source code that:
- Takes a string slice as input
- Returns a String as output
- Can use `std` library (no_std not required)
- Has access to common crates: `regex`, `serde_json`

### FR2: Compilation Pipeline

- Rust code is wrapped in a WASM-compatible template
- Compiled using `rustc` with `wasm32-unknown-unknown` target
- Compilation errors are captured and returned to LLM
- Successful compilation produces cached WASM module

### FR3: Sandboxed Execution

- Fuel limit: 10M instructions (configurable)
- Memory limit: 64MB (configurable)
- No filesystem access
- No network access
- No system calls beyond memory allocation

### FR4: Caching

- Compiled modules cached by source code hash
- Cache persists across RLM sessions
- Cache size limit with LRU eviction
- Cache hit returns pre-compiled module instantly

### FR5: Error Recovery

When compilation fails:
1. Error message returned to LLM
2. LLM can modify code and retry
3. Maximum 3 compilation attempts per iteration

## Non-Functional Requirements

### NFR1: Performance
- Compilation: < 5 seconds for typical functions
- Execution: < 1 second for most analyses
- Cache hit: < 10ms overhead

### NFR2: Security
- No sandbox escapes possible
- Resource exhaustion handled gracefully
- Malicious code cannot affect host system

### NFR3: Reliability
- Compilation failures don't crash RLM
- Execution failures return error strings
- Timeout handling for runaway code

## Success Metrics

1. **Adoption**: 30% of complex queries use WASM within 3 months
2. **Iteration Reduction**: Average iterations for complex queries reduced by 50%
3. **Token Savings**: 40% reduction in tokens for WASM-eligible queries
4. **Error Rate**: < 5% compilation failures after LLM retry

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| WASM sandbox escape | Critical | Low | Use proven wasmtime runtime, regular security audits |
| Compilation too slow | High | Medium | Aggressive caching, pre-warmed compiler |
| LLM generates broken code | Medium | High | Clear examples in prompt, retry mechanism |
| Resource exhaustion | Medium | Medium | Strict fuel/memory limits, timeout |

## Timeline

- **Phase 1** (2 weeks): Core compilation pipeline
- **Phase 2** (1 week): Caching layer
- **Phase 3** (1 week): LLM prompt engineering and examples
- **Phase 4** (1 week): Testing and hardening

## Appendix: Example Use Cases

### Example 1: Word Frequency Analysis
```rust
pub fn analyze(input: &str) -> String {
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for word in input.split_whitespace() {
        *counts.entry(word).or_insert(0) += 1;
    }
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1));
    pairs.iter().take(10)
        .map(|(w, c)| format!("{}: {}", w, c))
        .collect::<Vec<_>>()
        .join("\n")
}
```

### Example 2: JSON Field Extraction
```rust
pub fn analyze(input: &str) -> String {
    input.lines()
        .filter_map(|line| {
            serde_json::from_str::<serde_json::Value>(line).ok()
        })
        .filter_map(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .collect::<Vec<_>>()
        .join("\n")
}
```

## Follow-On: Multi-Input WASM (v2)

### Problem

The current `rust_wasm` command takes a single input (`on` field or the full context). However, many real-world analyses benefit from correlating multiple data sources:

- Compare error logs with configuration settings
- Cross-reference extracted data with lookup tables
- Combine results from previous iterations

Currently, the LLM must concatenate multiple variables with delimiters, then parse them apart inside the WASM code—error-prone and inefficient.

### Proposed Solution: Named Inputs

Extend `rust_wasm` to accept multiple named inputs that map to function parameters:

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(logs: &str, config: &str, errors: &str) -> String { ... }",
  "inputs": {
    "logs": "$log_excerpt",
    "config": "$config_data",
    "errors": "$error_lines"
  },
  "store": "correlation_result"
}
```

### How It Works

1. **Signature Parsing**: The compiler extracts parameter names from the function signature
2. **Input Mapping**: Each parameter name maps to a variable or literal in `inputs`
3. **Invocation**: WASM module called with arguments in parameter order
4. **Backward Compatible**: Omitting `inputs` falls back to single-input `on` field behavior

### Use Cases

#### Use Case 1: Log Correlation
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(errors: &str, timestamps: &str) -> String { let error_times: Vec<_> = errors.lines().zip(timestamps.lines()).filter(|(e, _)| e.contains(\"FATAL\")).map(|(_, t)| t).collect(); format!(\"Fatal errors at: {}\", error_times.join(\", \")) }",
  "inputs": {
    "errors": "$error_lines",
    "timestamps": "$timestamp_lines"
  },
  "store": "fatal_times"
}
```

#### Use Case 2: Config-Aware Analysis
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(data: &str, threshold: &str) -> String { let t: i64 = threshold.trim().parse().unwrap_or(100); let count = data.lines().filter(|l| l.parse::<i64>().map(|n| n > t).unwrap_or(false)).count(); format!(\"{} values exceed threshold {}\", count, t) }",
  "inputs": {
    "data": "$numeric_data",
    "threshold": "$user_threshold"
  },
  "store": "exceed_count"
}
```

#### Use Case 3: Multi-Document Comparison
```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(doc_a: &str, doc_b: &str) -> String { let words_a: std::collections::HashSet<_> = doc_a.split_whitespace().collect(); let words_b: std::collections::HashSet<_> = doc_b.split_whitespace().collect(); let common: Vec<_> = words_a.intersection(&words_b).collect(); format!(\"{} common words\", common.len()) }",
  "inputs": {
    "doc_a": "$section_1",
    "doc_b": "$section_2"
  },
  "store": "similarity"
}
```

### Implementation Notes

1. **WASM ABI**: Pass multiple string pointers via linear memory
2. **Parameter Limit**: Recommend max 5 inputs to keep signature manageable
3. **Type Safety**: All inputs are `&str`; numeric conversion happens in Rust code
4. **Caching**: Cache key includes input count, not input values

### Success Criteria

- LLM naturally uses multi-input when correlating stored variables
- 30% reduction in "glue code" iterations for complex analyses
- No performance regression vs. single-input mode

### Timeline Estimate

- **Phase 1**: Signature parsing and input mapping (~3 days)
- **Phase 2**: WASM invocation with multiple args (~2 days)
- **Phase 3**: System prompt updates and examples (~1 day)
- **Phase 4**: Testing and documentation (~2 days)

## Code Generation Approaches Architecture

This section documents all approaches considered for LLM-generated code execution.

### Capability Levels Architecture

RLM defines three capability levels for code execution, each with different tradeoffs between security and capability:

| Level | Name | Sandbox | Capabilities | Use Case |
|-------|------|---------|--------------|----------|
| 1 | DSL Operations | Full | Built-in find/regex/slice/count | Simple filtering, search |
| 2 | Sandboxed WASM | Full/Controlled | Map-reduce with controlled libs | Aggregation, counting |
| 3 | Native CLI/DLL | Process/None | Full Rust std library | Complex analysis, large data |

**Level 1: DSL Operations** (Most Secure)
- Built-in commands: find, regex, lines, slice, count
- No code generation - fixed command set
- Fully sandboxed - operates only on provided context
- Best for: Simple pattern matching, extraction, filtering

**Level 2: Sandboxed WASM** (Balanced)
- LLM generates code that runs in WASM sandbox
- Memory and execution limits enforced
- Option: Node.js WASM container with controlled library access
- Best for: Map-reduce patterns, frequency counting, aggregation
- Limitations: No HashMap, no complex string operations in wasmtime

**Level 3: Native CLI/DLL** (Maximum Capability)
- LLM generates full Rust code compiled to native binary
- Full std library: HashMap, contains, split, regex
- Process isolation only (CLI) or none (DLL)
- Future: Docker/LXC, seccomp, unprivileged user for sandboxing
- Best for: Complex analysis, large datasets, operations that fail in WASM

**Selection Guidance**:
1. Start with Level 1 (DSL) if the query can be answered with filtering/extraction
2. Use Level 2 (WASM) for aggregation on medium datasets (<100KB) with simple patterns
3. Fall back to Level 3 (CLI) for large datasets or when WASM fails

### 1. WASM-Based Approaches (Implemented)

#### 1.1 rust_wasm (Basic WASM)

**Command**: `{"op": "rust_wasm", "code": "...", "store": "result"}`

**Pros**:
- Sandboxed execution (no filesystem, network, process access)
- Memory limits enforced
- Fuel-based execution limits

**Cons**:
- WASM memory model causes crashes with complex string operations
- TwoWaySearcher (used by contains, find, split) panics in WASM
- HashMap/HashSet also cause panics
- Requires custom byte-level helpers to avoid panics

**Status**: Implemented but unreliable for complex operations

#### 1.2 rust_wasm_reduce_intent (Streaming WASM)

**Command**: `{"op": "rust_wasm_reduce_intent", "intent": "...", "store": "result"}`

**Pros**:
- Streaming pattern handles large datasets
- State persists across chunks
- Same sandbox protections as WASM

**Cons**:
- Same WASM limitations as rust_wasm
- State growing in WASM memory can cause crashes
- Complex aggregation still problematic

**Status**: Implemented but unreliable for large datasets

#### 1.3 rust_wasm_mapreduce (Stateless WASM Map + Native Reduce)

**Command**: `{"op": "rust_wasm_mapreduce", "intent": "...", "combiner": "count", "store": "result"}`

**Pros**:
- WASM is completely stateless (just maps lines to key-value pairs)
- All aggregation happens in native Rust (HashMap, sorting)
- More reliable than pure WASM approaches

**Cons**:
- Still subject to WASM string operation crashes
- Limited to map-reduce patterns

**Status**: Implemented, more reliable but still has WASM issues

### 2. Native CLI Binary Approach (Implemented)

#### 2.1 rust_cli_intent

**Command**: `{"op": "rust_cli_intent", "intent": "...", "store": "result"}`

**How It Works**:
1. Coding LLM generates Rust code with full std library access
2. Code compiled to native binary (cached by code hash)
3. Binary reads from stdin, writes to stdout
4. Process isolation provides sandboxing

**Pros**:
- Full Rust std library (HashMap, contains, split, etc.)
- No WASM memory limitations
- No TwoWaySearcher panics
- Faster execution for large datasets
- Binary caching for repeated queries

**Cons**:
- Less sandboxed than WASM (process isolation only)
- Relies on code validation to block dangerous operations

**Security**:
- Code validation blocks: std::fs::remove, std::fs::write, std::net, std::process::Command, unsafe blocks, extern crate
- Future: Docker/LXC containers, seccomp/landlock, unprivileged user execution

**Status**: Implemented, preferred for large datasets

### 3. Future Approaches (Considered but Not Yet Implemented)

#### 3.1 Dynamic cdylib (In-Process)

**Concept**: Compile Rust to a shared library (.so/.dylib), load dynamically, call in-process

**Pros**:
- No subprocess overhead
- Can share memory directly
- Potentially faster than CLI

**Cons**:
- Less isolation than separate process
- Crashes in loaded code crash the host
- More complex to implement safely

**Status**: Considered for future optimization

#### 3.2 Node.js WASM Container

**Concept**: Run WASM in a Node.js container with JavaScript glue code

**Pros**:
- V8's WASM runtime may have fewer limitations than wasmtime
- JavaScript can provide polyfills for problematic operations
- Container provides sandboxing

**Cons**:
- Requires Node.js dependency
- More complex architecture
- Still limited by WASM memory model

**Status**: Considered for future investigation

### 4. Capability Matrix

| Approach | HashMap | contains/split | Large Data | Sandbox | Speed |
|----------|---------|----------------|------------|---------|-------|
| rust_wasm | No | No (panics) | Limited | Full WASM | Medium |
| rust_wasm_reduce | No | No (panics) | Better | Full WASM | Medium |
| rust_wasm_mapreduce | Native | No (panics) | Good | Partial | Medium |
| rust_cli_intent | Yes | Yes | Excellent | Process | Fast |
| cdylib (future) | Yes | Yes | Excellent | None | Fastest |
| Node WASM (future) | Maybe | Maybe | Unknown | Container | Medium |

### 5. Recommendation

For production use:
1. **Small datasets (<100KB)**: Use rust_wasm_mapreduce with prefiltering
2. **Large datasets**: Use rust_cli_intent
3. **Security-critical**: Add Docker/seccomp sandboxing to CLI approach

### 6. Future Work: llm_chunk_analyze

**Concept**: For tasks requiring semantic understanding (not just computation), delegate to an LLM that processes chunks.

**Use Cases**:
- Categorization that requires reading/understanding
- Fuzzy matching that can't be done computationally
- Summarization of sections

**Architecture**:
```json
{"op": "llm_chunk_analyze", "chunk_size": 4096, "prompt": "Categorize each log entry...", "store": "categories"}
```

Each chunk sent to LLM with task prompt, results aggregated.

**Status**: Future feature for non-computational analysis
