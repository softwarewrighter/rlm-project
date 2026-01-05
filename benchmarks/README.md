# RLM Benchmarks

This directory contains test cases for measuring RLM effectiveness.

## Test Categories

### 1. NIAH (Needle in a Haystack)
Find specific facts hidden in large documents.

### 2. Log Analysis
Count, categorize, and summarize log entries.

### 3. Code Analysis
Find functions, classes, and patterns in source code.

## Running Tests

```bash
# Start the server (in one terminal)
cd rlm-orchestrator
DEEPSEEK_API_KEY="your-key" cargo run --bin rlm-server

# Run tests (in another terminal)
cd rlm-orchestrator
cargo run --bin rlm-test
```

The test runner will:
1. Find all JSON test files in `benchmarks/`
2. Generate contexts from specs
3. Run queries against the RLM server
4. Report pass/fail with timing stats

## Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | >80% | Correct answers vs ground truth |
| Iterations | <10 avg | Steps to reach answer |
| Token Usage | Measured | Total tokens sent/received |
| Latency | <30s | Time to answer |

## Adding Tests

Each test file is JSON:

```json
{
  "name": "test_name",
  "category": "niah|log|code",
  "context": "...",
  "queries": [
    {
      "query": "What is the secret code?",
      "expected": "ABC123",
      "match_type": "exact|contains|regex"
    }
  ]
}
```
