# RLM Benchmarks

This directory contains test cases for measuring RLM effectiveness.

## Test Categories

### 1. NIAH (Needle in a Haystack)
Find specific facts hidden in large documents.

### 2. Log Analysis
Count, categorize, and summarize log entries.

### 3. Code Analysis
Find functions, classes, and patterns in source code.

## Running Benchmarks

```bash
# Start the server
cd ../rlm-orchestrator
DEEPSEEK_API_KEY="your-key" cargo run --bin rlm-server &

# Run benchmarks
python run_benchmarks.py
```

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
