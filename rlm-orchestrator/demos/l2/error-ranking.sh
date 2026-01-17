#!/bin/bash
# Level 2 WASM: Error Type Frequency Demo
#
# Problem Type: OOLONG (frequency ranking)
# Ranks error types by frequency using WASM sandboxed computation.
# Uses rust_wasm_mapreduce with combiner="count" for frequency counting.
#
# Web UI Equivalent: "Rank errors by frequency" dropdown
# Expected time: 15-25 seconds
# Expected iterations: 1-2

source "$(dirname "$0")/../common.sh"

echo "============================================"
echo "Level 2 WASM: Error Type Frequency"
echo "============================================"
echo ""
echo "Problem Type: OOLONG (frequency ranking)"
echo "This demo ranks error types by frequency using WASM."
echo "Uses rust_wasm_mapreduce with combiner='count'."
echo ""

# Generate log data with various error types
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

echo "Generating 200 log entries..."
for i in $(seq 1 200); do
    h=$((10 + i / 60))
    m=$((i % 60))
    s=$((i % 60))
    ip="192.168.1.$((100 + i % 50))"

    # Generate different error types with varying frequencies
    type_idx=$((i % 10))
    case $type_idx in
        0|1|2) level="ERROR"; msg="AuthenticationFailed" ;;  # 30%
        3|4) level="ERROR"; msg="DatabaseError" ;;           # 20%
        5) level="ERROR"; msg="NetworkTimeout" ;;            # 10%
        6) level="WARN"; msg="SlowQuery" ;;
        7) level="WARN"; msg="HighMemory" ;;
        *) level="INFO"; msg="RequestProcessed" ;;
    esac

    printf "2024-01-15 %02d:%02d:%02d [%s] %s from %s\n" \
        $h $m $s $level $msg $ip >> "$CONTEXT_FILE"
done

lines=$(wc -l < "$CONTEXT_FILE" | tr -d ' ')
echo "Generated: $lines log entries"

# Query matches Web UI example
QUERY="Rank the error types from most to least frequent"

run_demo "$CONTEXT_FILE" "$QUERY" -vv
