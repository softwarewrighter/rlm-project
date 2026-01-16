#!/bin/bash
# Run all simple tests and validate results
# Usage: ./run-all.sh

# Don't exit on error - we want to run all tests
cd "$(dirname "$0")/../.."

# Source environment
export $(cat ~/.env | grep -v '^#' | xargs)

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local data="$1"
    local query="$2"
    local expected="$3"
    local timeout_sec="${4:-60}"

    echo ""
    echo "================================================================"
    echo "TEST: $query"
    echo "DATA: $data"
    echo "EXPECTED: $expected"
    echo "================================================================"

    # Run the test and capture output
    OUTPUT=$(timeout "$timeout_sec" ./target/release/rlm "$data" "$query" \
        --litellm \
        -m deepseek/deepseek-coder \
        2>&1) || true

    # Extract the answer (content between the two ════ border lines after "Answer:")
    ANSWER=$(echo "$OUTPUT" | awk '/^Answer:$/{found=1; next} found && /^════/{count++; if(count==2) exit; next} found{print}' | tr '\n' ' ' | xargs)

    echo "ANSWER: $ANSWER"

    # Check if output contains expected pattern
    if echo "$OUTPUT" | grep -q "$expected"; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}FAILED${NC}"
        echo "Full output:"
        echo "$OUTPUT"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

echo "RLM Simple Test Suite"
echo "===================="
echo "Using LiteLLM @ localhost:4000"
echo "Model: deepseek/deepseek-coder"

# Test 1: Count lines
run_test "tests/simple/data-5lines.txt" \
    "How many lines are there?" \
    "5"

# Test 2: Sum numbers
run_test "tests/simple/data-numbers.txt" \
    "What is the sum of all numbers?" \
    "150"

# Test 3: Count ERROR lines
run_test "tests/simple/data-logs.txt" \
    "How many ERROR lines are there?" \
    "5"

# Test 4: Rank error types
run_test "tests/simple/data-logs.txt" \
    "List the error types and their counts" \
    "ConnectionFailed"

# Test 5: IP extraction and ranking (demo pattern)
run_test "tests/simple/data-ips.txt" \
    "How many unique IP addresses are there? List each IP with its count." \
    "192.168.1.10" \
    90

# Test 6: Percentile/median calculation (demo pattern)
run_test "tests/simple/data-response-times.txt" \
    "Each line ends with a response time like '15ms'. Extract all response time numbers, sort them, and calculate the median." \
    "25" \
    90

# Test 7: Name counting (demo pattern for character analysis)
run_test "tests/simple/data-characters.txt" \
    "Count occurrences of each surname: Smith, Brown, Wilson. Each appears multiple times in the text." \
    "Smith" \
    90

echo ""
echo "================================================================"
echo "RESULTS: $TESTS_PASSED passed, $TESTS_FAILED failed"
echo "================================================================"

if [ $TESTS_FAILED -gt 0 ]; then
    exit 1
fi
