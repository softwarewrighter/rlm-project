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
