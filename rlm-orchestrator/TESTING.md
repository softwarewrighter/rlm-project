# RLM Testing Guide

## Important: Use LiteLLM Only

**Do NOT use local Ollama models directly.** Always use the LiteLLM gateway to access DeepSeek models.

## Environment Setup

### Required Environment Variables

All keys are stored in `~/.env`. **Always source this file before running:**

```bash
export $(cat ~/.env | grep -v '^#' | xargs)
```

The file contains:
```
LITELLM_HOST=localhost:4000
LITELLM_KEY=sk-local-test-key-123
LITELLM_MASTER_KEY=sk-local-test-key-123
LITELLM_API_KEY=sk-local-test-key-123
DEEPSEEK_API_KEY=sk-...
```

### LiteLLM Gateway

We use LiteLLM as a gateway to DeepSeek APIs.

- **Gateway URL**: `http://localhost:4000`
- **Models available**:
  - `deepseek/deepseek-chat` - General reasoning (base LLM)
  - `deepseek/deepseek-coder` - Code generation (codegen LLM for rust_wasm_intent)

## CLI Usage

The CLI auto-configures codegen to use LiteLLM when `--litellm` flag is set.

### Basic Usage

```bash
# Source environment first!
export $(cat ~/.env | grep -v '^#' | xargs)

# Simple query (auto-configures codegen via LiteLLM)
./target/release/rlm data.txt "Your query" --litellm -m deepseek/deepseek-coder

# Verbose output
./target/release/rlm data.txt "Your query" --litellm -m deepseek/deepseek-coder -v

# Extra verbose (shows commands)
./target/release/rlm data.txt "Your query" --litellm -m deepseek/deepseek-coder -vv
```

### CLI Flags

```bash
# LiteLLM configuration
--litellm                   # Enable LiteLLM mode (REQUIRED)
--litellm-url <URL>         # LiteLLM URL (default: http://localhost:4000)
-m, --model <MODEL>         # Model name (e.g., deepseek/deepseek-coder)

# Code generation (auto-configured when --litellm is set)
--codegen-model <MODEL>     # Override codegen model (default: same as --model)

# Verbosity
-v                          # Verbose output
-vv                         # Extra verbose (show commands)
--no-rust-wasm              # Disable rust_wasm commands
```

## Running Tests

### Simple Test Suite

```bash
cd /Users/mike/github/softwarewrighter/rlm-project/rlm-orchestrator

# Source environment
export $(cat ~/.env | grep -v '^#' | xargs)

# Run all 7 simple tests (~5-7 minutes total)
./tests/simple/run-all.sh
```

### Individual Test

```bash
export $(cat ~/.env | grep -v '^#' | xargs)

./target/release/rlm tests/simple/data-5lines.txt \
    "How many lines are there?" \
    --litellm \
    -m deepseek/deepseek-coder \
    -v
```

## Running Demos

### Start Server First

```bash
cd /Users/mike/github/softwarewrighter/rlm-project/rlm-orchestrator

# Source environment
export $(cat ~/.env | grep -v '^#' | xargs)

# Kill any existing server
pkill -9 -f rlm-server 2>/dev/null

# Start server
./target/release/rlm-server config.toml &>/tmp/rlm-server.log &

# Verify
curl -s http://localhost:8080/health
```

### Run a Demo

```bash
# Large logs error ranking
./demos/large-logs-error-ranking.sh

# War and Peace character analysis
./demos/war-and-peace-family-tree.sh
```

## Test Cases

### Level 1: Basic DSL Commands

Simple tests that complete with basic DSL commands (find, count, lines):

1. **Count lines**: "How many lines are there?"
2. **Find text**: "Find lines containing 'apple'"
3. **Count matches**: "How many lines contain 'berry'?"

### Level 2: Code Generation (rust_wasm_intent)

Tests that require code generation via the helper LLM:

1. **Sum numbers**: "Sum all the numbers"
2. **Count occurrences**: "Count occurrences of 'Smith', 'Brown', 'Wilson'"
3. **Calculate median**: "Calculate the median of all numbers"

### Level 3: Complex Analysis

1. **Frequency count and rank**: "Rank error types by frequency"
2. **IP extraction and counting**: "List unique IPs with their counts"

## Troubleshooting

### "No LiteLLM API key found"

Make sure you sourced the env file:
```bash
export $(cat ~/.env | grep -v '^#' | xargs)
echo $LITELLM_MASTER_KEY  # Should show the key
```

### "Code generation LLM error: Provider ret..."

This usually means the LiteLLM gateway timed out or the codegen model is overloaded. Try again or check:
```bash
curl -s http://localhost:4000/health
```

### WASM Panics (TwoWaySearcher, memcmp)

The generated code is using forbidden operations. The code generator should avoid these, but if you see these errors, the LLM generated unsafe code.

Forbidden operations:
- `.contains()` - use `has()` instead
- `.find()` - use `after()`/`before()` instead
- `.split()` - use `word()` instead
- `HashMap`/`HashSet` - use `Vec<(String, usize)>` instead

### Server Won't Start

Check the logs:
```bash
cat /tmp/rlm-server.log
```

Common issues:
- Missing API keys
- No providers configured
- Port already in use
