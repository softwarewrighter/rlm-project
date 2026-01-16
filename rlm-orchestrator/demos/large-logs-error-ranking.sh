#!/bin/bash
# Large Logs Error Ranking Demo
#
# Analyzes 5000 log lines to rank error types by frequency.
# Uses rust_wasm with HashMap for efficient aggregation.
#
# Expected time with DeepSeek: 30-60 seconds
# Expected iterations: 2-3

source "$(dirname "$0")/common.sh"

echo "============================================"
echo "Large Logs Error Ranking Demo"
echo "============================================"
echo ""
echo "This demo analyzes 5000 log lines and ranks error types"
echo "from most to least frequent using WASM HashMap."
echo ""

# Create temp file for context
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

# Fetch the sample data
fetch_sample "/samples/large-logs" "$CONTEXT_FILE"

# Run the query
QUERY="Rank the error types from most to least frequent. Show the count for each error type."

run_demo "$CONTEXT_FILE" "$QUERY" -vv
