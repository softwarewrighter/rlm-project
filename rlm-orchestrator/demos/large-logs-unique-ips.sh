#!/bin/bash
# Large Logs Unique IPs Demo
#
# Analyzes 5000 log lines to count and rank unique IP addresses.
# Uses rust_cli_intent for native binary code generation.
#
# Expected time with DeepSeek: 30-60 seconds
# Expected iterations: 1-2

source "$(dirname "$0")/common.sh"

echo "============================================"
echo "Large Logs Unique IPs Demo"
echo "============================================"
echo ""
echo "This demo analyzes 5000 log lines to find unique IPs"
echo "and rank the top 10 most active using native CLI code generation."
echo ""

# Create temp file for context
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

# Fetch the sample data
fetch_sample "/samples/large-logs" "$CONTEXT_FILE"

# Run the query
QUERY="How many unique IP addresses appear in these logs? List the top 10 most active IPs."

run_demo "$CONTEXT_FILE" "$QUERY" -vv
