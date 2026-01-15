#!/bin/bash
# Demo: IP Address Analysis with RLM + WASM
#
# This demo shows RLM using rust_wasm to analyze IP addresses in server logs.
# The WASM approach enables custom analysis that would be inefficient with
# simple text commands.
#
# Prerequisites:
#   - Ollama running locally with llama3.2:3b model
#   - cargo build --release in rlm-orchestrator directory
#
# Usage:
#   ./demo-ip-analysis.sh           # Default verbosity
#   ./demo-ip-analysis.sh -v        # Show LLM calls and timing
#   ./demo-ip-analysis.sh -vv       # Show full responses and commands

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/sample.log"

cd "$PROJECT_DIR/rlm-orchestrator" || exit 1

echo "Running RLM IP analysis demo..."
echo "Query: Find IP addresses making more than 3 requests"
echo ""

cargo run --release --bin rlm -- \
    "$LOG_FILE" \
    "Find all IP addresses that appear more than 3 times in the logs and list them with their counts" \
    "$@"
