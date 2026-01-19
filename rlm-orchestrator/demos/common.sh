#!/bin/bash
# Common setup for RLM demos
# Sources API keys from ~/.env and sets up paths
# Uses DeepSeek via LiteLLM gateway by default

set -e

# Load environment variables from ~/.env
if [ -f ~/.env ]; then
    set -a
    source ~/.env
    set +a
else
    echo "Warning: ~/.env not found. API keys may not be set."
fi

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Path to the RLM CLI binary
RLM_BIN="$PROJECT_ROOT/target/release/rlm"

# Check if binary exists
if [ ! -f "$RLM_BIN" ]; then
    echo "Error: RLM binary not found at $RLM_BIN"
    echo "Run 'cargo build --release' first."
    exit 1
fi

# Server URL for fetching sample data
RLM_SERVER="${RLM_SERVER:-http://localhost:8080}"

# LiteLLM gateway settings (for all LLM access)
LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"
LITELLM_MODEL="${LITELLM_MODEL:-deepseek/deepseek-chat}"

# Code generation LLM settings (also via LiteLLM)
# Uses deepseek-coder for rust_cli_intent code generation
CODEGEN_MODEL="${CODEGEN_MODEL:-deepseek/deepseek-coder}"

# Function to fetch sample data from the server
fetch_sample() {
    local sample_path="$1"
    local output_file="$2"

    echo "Fetching sample data from $RLM_SERVER$sample_path..."
    curl -s "$RLM_SERVER$sample_path" > "$output_file"

    if [ ! -s "$output_file" ]; then
        echo "Error: Failed to fetch sample data. Is the server running?"
        echo "Start with: cargo run --bin rlm-server -- config.toml"
        exit 1
    fi

    local size=$(wc -c < "$output_file" | tr -d ' ')
    local lines=$(wc -l < "$output_file" | tr -d ' ')
    echo "Downloaded: $lines lines, $size bytes"
}

# Function to run a demo with timing
# Uses DeepSeek via LiteLLM gateway for both base and codegen LLMs
# Usage: run_demo <context_file> <query> [extra_flags...]
run_demo() {
    local context_file="$1"
    local query="$2"
    shift 2  # Remove first two args, rest are extra flags

    echo ""
    echo "Query: $query"
    echo "Model: $LITELLM_MODEL (via LiteLLM @ $LITELLM_URL)"
    echo "CodeGen: $CODEGEN_MODEL (via LiteLLM @ $LITELLM_URL)"
    echo ""
    echo "Running..."
    echo "----------------------------------------"

    time "$RLM_BIN" "$context_file" "$query" "$@" \
        --litellm \
        --litellm-url "$LITELLM_URL" \
        --model "$LITELLM_MODEL" \
        --codegen-model "$CODEGEN_MODEL"
}
