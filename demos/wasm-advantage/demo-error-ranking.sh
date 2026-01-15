#!/bin/bash
# Demo: Error Ranking with RLM
#
# This demo shows RLM analyzing error types in server logs and ranking them
# by frequency. This is a more complex query that requires the LLM to
# identify, categorize, count, and sort errors.
#
# Prerequisites:
#   - Ollama running locally with llama3.2:3b model
#   - cargo build --release in rlm-orchestrator directory
#
# Usage:
#   ./demo-error-ranking.sh         # Default verbosity
#   ./demo-error-ranking.sh -v      # Show LLM calls and timing
#   ./demo-error-ranking.sh -vv     # Show full responses and commands

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_FILE="$SCRIPT_DIR/sample.log"

cd "$PROJECT_DIR/rlm-orchestrator" || exit 1

echo "Running RLM error ranking demo..."
echo "Query: Rank errors from most to least frequent"
echo ""

cargo run --release --bin rlm -- \
    "$LOG_FILE" \
    "rank the errors from most often to least often found in the logs" \
    "$@"
