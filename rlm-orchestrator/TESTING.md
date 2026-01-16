# RLM Testing Guide

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

We use LiteLLM as a gateway to DeepSeek APIs. **Do not use local Ollama models** - they're not good enough.

- **Gateway URL**: `http://localhost:4000`
- **Models available**:
  - `deepseek/deepseek-chat` - General reasoning
  - `deepseek/deepseek-coder` - Code generation (use for both base and helper LLMs)

## Configuration

### config.toml

The server uses `config.toml` in the project root:

```toml
# Disable bypass to ensure RLM is always used (for testing)
bypass_enabled = false
bypass_threshold = 0

# WASM configuration
[wasm]
enabled = true
rust_wasm_enabled = true

# Code generation via LiteLLM gateway
codegen_provider = "litellm"
codegen_url = "http://localhost:4000"
codegen_model = "deepseek/deepseek-coder"

# LLM provider via LiteLLM
[[providers]]
provider_type = "litellm"
base_url = "http://localhost:4000"
model = "deepseek/deepseek-coder"
role = "both"
weight = 1
```

### CLI Flags

The CLI has its own configuration that overrides config.toml:

```bash
# Use LiteLLM for base LLM
--litellm                   # Enable LiteLLM mode
--litellm-url <URL>         # LiteLLM URL (default: http://localhost:4000)
--litellm-key <KEY>         # API key (or use LITELLM_MASTER_KEY env var)
-m, --model <MODEL>         # Model name (e.g., deepseek/deepseek-coder)

# Code generation (rust_wasm_intent command)
--codegen-url <URL>         # Codegen LLM URL
--codegen-model <MODEL>     # Codegen model name

# Other
-v                          # Verbose output
-vv                         # Extra verbose (show commands)
--no-rust-wasm              # Disable rust_wasm commands
```

## Running Tests

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

### Run CLI Tests

```bash
cd /Users/mike/github/softwarewrighter/rlm-project/rlm-orchestrator

# Source environment
export $(cat ~/.env | grep -v '^#' | xargs)

# Simple test with LiteLLM
./target/release/rlm tests/simple/data-5lines.txt \
    "How many lines are there?" \
    --litellm \
    -m deepseek/deepseek-coder \
    --codegen-url http://localhost:4000 \
    --codegen-model deepseek/deepseek-coder \
    -v
```

## Test Cases

### Level 1: No WASM Required

Simple tests that should complete with basic DSL commands (find, count, lines):

1. **Count lines**: "How many lines are there?"
2. **Find text**: "Find lines containing 'apple'"
3. **Count matches**: "How many lines contain 'berry'?"

### Level 2: Simple WASM (rust_wasm_intent)

Tests that require code generation:

1. **Sum numbers**: "Sum all the numbers"
2. **Extract pattern**: "Extract all words starting with 'a'"

### Level 3: Complex Analysis

1. **Frequency count**: "Count occurrences of each word"
2. **Sort and rank**: "Rank items by frequency"

## Troubleshooting

### "No LiteLLM API key found"

Make sure you sourced the env file:
```bash
export $(cat ~/.env | grep -v '^#' | xargs)
echo $LITELLM_MASTER_KEY  # Should show the key
```

### "Code generation LLM not configured"

Make sure you passed `--codegen-url`:
```bash
--codegen-url http://localhost:4000 --codegen-model deepseek/deepseek-coder
```

### WASM Panics (TwoWaySearcher, memcmp)

The generated code is using forbidden operations. Check:
1. Using `has()` instead of `.contains()`
2. Using `eq()` instead of `==` for strings
3. Using `Vec<(String, usize)>` instead of `HashMap`

### Server Won't Start

Check the config:
```bash
cat /tmp/rlm-server.log
```

Common issues:
- Missing API keys
- No providers configured
- Port already in use
