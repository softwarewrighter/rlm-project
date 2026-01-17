#!/bin/bash
# Level 2 WASM: Unique IP Count Demo
#
# Problem Type: OOLONG (aggregation)
# Counts unique IP addresses in log data using WASM sandboxed computation.
# Uses rust_wasm_mapreduce with combiner="unique" for O(1) HashSet operations.
#
# Web UI Equivalent: "Count unique IP addresses" dropdown
# Expected time: 15-25 seconds
# Expected iterations: 1-2

source "$(dirname "$0")/../common.sh"

echo "============================================"
echo "Level 2 WASM: Unique IP Count"
echo "============================================"
echo ""
echo "Problem Type: OOLONG (aggregation)"
echo "This demo counts unique IP addresses using WASM."
echo "Uses rust_wasm_mapreduce with combiner='unique'."
echo ""

# Generate log data with IPs
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

echo "Generating 200 log entries..."
for i in $(seq 1 200); do
    h=$((10 + i / 60))
    m=$((i % 60))
    s=$((i % 60))
    # Use 8 different IPs to match Web UI data
    ip_idx=$((i % 8))
    case $ip_idx in
        0) ip="192.168.1.100" ;;
        1) ip="10.0.0.50" ;;
        2) ip="172.16.0.25" ;;
        3) ip="10.0.0.75" ;;
        4) ip="192.168.1.200" ;;
        5) ip="172.16.0.30" ;;
        6) ip="10.0.0.60" ;;
        7) ip="192.168.1.105" ;;
    esac

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

    printf "2024-01-15 %02d:%02d:%02d [%s] %s from %s - Request to /api/users\n" \
        $h $m $s $level $msg $ip >> "$CONTEXT_FILE"
done

lines=$(wc -l < "$CONTEXT_FILE" | tr -d ' ')
echo "Generated: $lines log entries"

# Query matches Web UI example
QUERY="How many unique IP addresses are in these logs?"

run_demo "$CONTEXT_FILE" "$QUERY" -vv
