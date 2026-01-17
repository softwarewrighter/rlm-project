#!/bin/bash
# Basic Error Count Demo
#
# Simple demo that counts ERROR lines in log data.
# Uses basic RLM commands (no WASM) to demonstrate the core approach.
#
# Expected time with DeepSeek: 15-30 seconds
# Expected iterations: 1-2

source "$(dirname "$0")/../common.sh"

echo "============================================"
echo "Basic Error Count Demo"
echo "============================================"
echo ""
echo "This demo counts ERROR lines using basic RLM commands."
echo "Good for understanding the RLM approach before WASM demos."
echo ""

# Generate log data
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

echo "Generating 100 log entries..."
for i in $(seq 1 100); do
    h=$((10 + i / 60))
    m=$((i % 60))
    s=$((i % 60))
    ip="192.168.1.$((100 + i % 100))"

    if [ $((i % 5)) -eq 0 ]; then
        level="ERROR"
        msg="AuthenticationFailed"
    elif [ $((i % 3)) -eq 0 ]; then
        level="WARN"
        msg="SlowQuery"
    else
        level="INFO"
        msg="RequestProcessed"
    fi

    printf "[2024-01-15 %02d:%02d:%02d] %s %s from %s\n" \
        $h $m $s $level $msg $ip >> "$CONTEXT_FILE"
done

lines=$(wc -l < "$CONTEXT_FILE" | tr -d ' ')
echo "Generated: $lines log entries"

# Run the query
QUERY="How many ERROR lines are there?"

run_demo "$CONTEXT_FILE" "$QUERY" -vv
