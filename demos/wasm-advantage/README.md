# WASM Advantage Demo

This demo shows why `rust_wasm` is powerful compared to built-in RLM commands.

## The Challenge

Given the log file `sample.log`, answer this question:

> **"What are the top 5 error types by frequency, and what's the average time between consecutive errors of each type?"**

This requires:
1. Pattern extraction (error type names)
2. Frequency counting (HashMap aggregation)
3. Timestamp parsing (to compute time deltas)
4. Math operations (averaging)

## Built-in Commands: The Struggle

With only `find`, `regex`, `slice`, `lines`, `count`, and `llm_query`, here's what we'd need:

```
Iteration 1: regex "ERROR.*" -> find all error lines
Iteration 2: Store the result
Iteration 3: Ask LLM to categorize error types (unreliable)
Iteration 4: For each type, count occurrences
Iteration 5-N: Multiple passes to count each type
...can't compute time between errors - no timestamp parsing!
...can't compute averages - no math operations!
```

**Result:** 5-10+ iterations, still can't do the timestamp/math parts!

## rust_wasm: One Pass Solution

```json
{
  "op": "rust_wasm",
  "code": "pub fn analyze(input: &str) -> String { use std::collections::HashMap; let mut error_times: HashMap<String, Vec<u64>> = HashMap::new(); for line in input.lines() { if !line.contains(\"ERROR\") { continue; } let parts: Vec<&str> = line.splitn(4, ' ').collect(); if parts.len() < 4 { continue; } let time_str = format!(\"{} {}\", parts[0], parts[1]); let mins: u64 = time_str.split(':').nth(1).and_then(|m| m.parse().ok()).unwrap_or(0); let secs: u64 = time_str.split(':').nth(2).and_then(|s| s.split('.').next()).and_then(|s| s.parse().ok()).unwrap_or(0); let timestamp = mins * 60 + secs; let error_type = if let Some(bracket_end) = parts[3].find(']') { let after_bracket = &parts[3][bracket_end+2..]; after_bracket.split(':').next().unwrap_or(\"Unknown\").to_string() } else { \"Unknown\".to_string() }; error_times.entry(error_type).or_default().push(timestamp); } let mut results: Vec<(String, usize, f64)> = error_times.iter().map(|(etype, times)| { let count = times.len(); let avg_gap = if times.len() > 1 { let mut sorted = times.clone(); sorted.sort(); let gaps: Vec<u64> = sorted.windows(2).map(|w| w[1] - w[0]).collect(); gaps.iter().sum::<u64>() as f64 / gaps.len() as f64 } else { 0.0 }; (etype.clone(), count, avg_gap) }).collect(); results.sort_by(|a, b| b.1.cmp(&a.1)); results.iter().take(5).map(|(t, c, g)| format!(\"{}: {} occurrences, avg {:.1}s between\", t, c, g)).collect::<Vec<_>>().join(\"\\n\") }",
  "store": "error_analysis"
}
```

## Running the Demo

### Option 1: Using the CLI

```bash
cd rlm-orchestrator
cargo run --bin rlm -- ../demos/wasm-advantage/sample.log \
  "What are the top 5 error types by frequency, and what's the average time between consecutive errors of each type?" \
  -vv
```

### Option 2: Manual Test

You can test just the WASM execution with:

```bash
cargo test test_demo_error_analysis -- --nocapture
```

## Expected Output

```
ConnectionTimeout: 9 occurrences, avg 147.5s between
AuthenticationFailed: 12 occurrences, avg 98.2s between
ValidationError: 8 occurrences, avg 135.0s between
ConnectionRefused: 3 occurrences, avg 240.0s between
DeadlockDetected: 2 occurrences, avg 839.0s between
```

## Why This Matters

| Capability | Built-in Commands | rust_wasm |
|------------|-------------------|-----------|
| Find errors | Yes (regex) | Yes |
| Count total errors | Yes (count) | Yes |
| Group by error type | No (needs LLM guessing) | Yes (HashMap) |
| Parse timestamps | No | Yes |
| Calculate time deltas | No | Yes |
| Compute averages | No | Yes |
| Sort by frequency | No | Yes |
| **Do it all in one pass** | **No** | **Yes** |

## The Code Explained

The rust_wasm code does:

1. **Filter**: Only process lines containing "ERROR"
2. **Parse timestamp**: Extract minutes and seconds from log timestamp
3. **Extract error type**: Parse the error category (ConnectionTimeout, AuthenticationFailed, etc.)
4. **Aggregate**: Store timestamps by error type in a HashMap
5. **Calculate gaps**: For each type, compute time between consecutive occurrences
6. **Average**: Calculate mean gap time
7. **Sort**: Order by frequency (most common first)
8. **Format**: Return top 5 as readable output

All in a single RLM iteration, where built-in commands would need many iterations and still couldn't complete the task.
