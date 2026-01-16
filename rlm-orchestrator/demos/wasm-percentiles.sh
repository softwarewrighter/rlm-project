#!/bin/bash
# CLI Percentiles Demo
#
# Calculates response time percentiles (p50, p95, p99) from log data.
# Uses rust_cli_intent to parse times, sort, and compute percentiles.
#
# Expected time with DeepSeek: 20-40 seconds
# Expected iterations: 2-3

source "$(dirname "$0")/common.sh"

echo "============================================"
echo "CLI Percentiles Demo"
echo "============================================"
echo ""
echo "This demo calculates p50, p95, and p99 response time"
echo "percentiles using native CLI code generation."
echo ""

# Generate response time log data
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

echo "Generating 300 response time log entries..."
for i in $(seq 1 300); do
    # Generate realistic response times (most fast, some slow)
    if [ $((i % 20)) -eq 0 ]; then
        time=$((500 + RANDOM % 1500))  # Slow responses (500-2000ms)
    elif [ $((i % 5)) -eq 0 ]; then
        time=$((100 + RANDOM % 400))   # Medium responses (100-500ms)
    else
        time=$((10 + RANDOM % 90))     # Fast responses (10-100ms)
    fi
    printf "2024-01-15 %02d:%02d:%02d GET /api/endpoint %dms\n" \
        $((10 + i / 60)) $((i % 60)) $((i % 60)) $time >> "$CONTEXT_FILE"
done

lines=$(wc -l < "$CONTEXT_FILE" | tr -d ' ')
echo "Generated: $lines log entries"

# Run the query
QUERY="Calculate the p50, p95, and p99 response time percentiles from these logs."

run_demo "$CONTEXT_FILE" "$QUERY" -vv
