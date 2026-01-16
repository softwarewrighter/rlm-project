#!/bin/bash
# Simple test runner - sources ~/.env and runs a single RLM query
# Usage: ./run-test.sh <data-file> "<query>" [expected-pattern]

set -e
cd "$(dirname "$0")/../.."

# Source environment
export $(cat ~/.env | grep -v '^#' | xargs)

DATA="$1"
QUERY="$2"
EXPECTED="$3"

if [[ -z "$DATA" || -z "$QUERY" ]]; then
    echo "Usage: $0 <data-file> \"<query>\" [expected-pattern]"
    exit 1
fi

echo "=== Test: $QUERY ==="
echo "Data: $DATA"
echo ""

# Run with verbose output, timeout 60s
timeout 60 ./target/release/rlm "$DATA" "$QUERY" -v 2>&1

RESULT=$?

if [[ -n "$EXPECTED" ]]; then
    echo ""
    echo "--- Checking for expected pattern: $EXPECTED ---"
    # This is a placeholder - actual verification would need the output
fi

exit $RESULT
