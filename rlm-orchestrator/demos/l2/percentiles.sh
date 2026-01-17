#!/bin/bash
# Level 2 WASM: Response Time Percentiles Demo
#
# Problem Type: OOLONG (statistical computation)
# Calculates p50, p95, p99 response time percentiles using WASM.
# Uses rust_wasm_intent (not mapreduce) because percentiles need sorted data.
#
# Web UI Equivalent: "Response time percentiles" dropdown
# Expected time: 15-30 seconds
# Expected iterations: 1-2

source "$(dirname "$0")/../common.sh"

echo "============================================"
echo "Level 2 WASM: Response Time Percentiles"
echo "============================================"
echo ""
echo "Problem Type: OOLONG (statistical computation)"
echo "This demo calculates p50, p95, p99 percentiles using WASM."
echo "Uses rust_wasm_intent (needs sorted data, not per-line mapreduce)."
echo ""

# Generate response time log data
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

echo "Generating 300 response time log entries..."
for i in $(seq 1 300); do
    # Generate realistic response times (most fast, some slow outliers)
    if [ $((i % 20)) -eq 0 ]; then
        time=$((500 + RANDOM % 1500))  # Slow responses (500-2000ms) - 5%
    elif [ $((i % 5)) -eq 0 ]; then
        time=$((100 + RANDOM % 400))   # Medium responses (100-500ms) - 15%
    else
        time=$((10 + RANDOM % 90))     # Fast responses (10-100ms) - 80%
    fi

    endpoints=("/api/users" "/api/data" "/api/health" "/api/products" "/api/orders")
    ep=${endpoints[$((i % 5))]}
    methods=("GET" "POST" "PUT" "DELETE")
    method=${methods[$((i % 4))]}

    printf "%s %s - %dms\n" "$method" "$ep" "$time" >> "$CONTEXT_FILE"
done

lines=$(wc -l < "$CONTEXT_FILE" | tr -d ' ')
echo "Generated: $lines log entries"

# Query matches Web UI example - explicit guidance for WASM intent
QUERY="Calculate the p50, p95, and p99 response time percentiles. Use rust_wasm_intent with intent 'Extract the number before ms from each line, sort all numbers, and compute p50 (median), p95, and p99 percentiles'."

run_demo "$CONTEXT_FILE" "$QUERY" -vv
