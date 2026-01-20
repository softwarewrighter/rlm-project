#!/bin/bash
# Level 3 CLI: Response Time Percentiles Demo
# Problem Type: OOLONG (Long-context reasoning)
#
# This demo shows how L1+L3 hybrid approach works for statistics:
# 1. L1 DSL commands (regex) extract response times from log lines
# 2. L3 rust_cli_intent computes p50, p95, p99 percentiles
#
# This hybrid approach:
# - L1 regex extracts just the millisecond values (simpler input)
# - L3 CLI uses Vec<f64> and sorting for percentile calculation
# - Native code handles sorting and indexing efficiently

set -e
cd "$(dirname "$0")/../.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TEMP_FILE="/tmp/response-times.txt"
QUERY="Calculate the p50, p95, and p99 response time percentiles."
SERVER_URL="${SERVER_URL:-http://localhost:4539}"

clear
echo ""
echo "================================================================"
echo -e "${BLUE}    Level 3 CLI Demo: Response Time Percentiles${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Problem Type:${NC} OOLONG"
echo -e "${CYAN}Dataset:${NC} 2000 response time entries"
echo -e "${CYAN}Strategy:${NC} L1 regex extracts times, L3 CLI computes percentiles"
echo -e "${CYAN}Query:${NC} $QUERY"
echo ""

# Check if RLM server is running
if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}RLM server not running. Start it with:${NC}"
    echo ""
    echo "  cd rlm-orchestrator"
    echo "  cargo run --bin rlm-server -- --enable-cli"
    echo ""
    exit 1
fi

echo -e "${YELLOW}Fetching response time dataset...${NC}"
curl -s "$SERVER_URL/samples/response-times" > "$TEMP_FILE"
LINE_COUNT=$(wc -l < "$TEMP_FILE")
CHAR_COUNT=$(wc -c < "$TEMP_FILE")
echo -e "${GREEN}Downloaded:${NC} $LINE_COUNT lines, $CHAR_COUNT bytes"
echo ""

# Show sample of data
echo -e "${YELLOW}Sample response times:${NC}"
head -5 "$TEMP_FILE"
echo "..."
echo ""

# Show distribution info
echo -e "${YELLOW}Response time distribution:${NC}"
grep -oE '[0-9]+ms' "$TEMP_FILE" | sed 's/ms//' | sort -n > /tmp/times-sorted.txt
P50_LINE=$((LINE_COUNT / 2))
P95_LINE=$((LINE_COUNT * 95 / 100))
P99_LINE=$((LINE_COUNT * 99 / 100))
P50=$(sed -n "${P50_LINE}p" /tmp/times-sorted.txt)
P95=$(sed -n "${P95_LINE}p" /tmp/times-sorted.txt)
P99=$(sed -n "${P99_LINE}p" /tmp/times-sorted.txt)
echo -e "  Ground truth: p50=${P50}ms, p95=${P95}ms, p99=${P99}ms"
rm -f /tmp/times-sorted.txt
echo ""

echo "================================================================"
echo -e "${GREEN}Running RLM with CLI enabled...${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Expected workflow:${NC}"
echo "  1. L1: regex to extract millisecond values from each line"
echo "  2. L3: rust_cli_intent to parse, sort, and compute percentiles"
echo "  3. Return p50, p95, p99 values"
echo ""

# Run RLM CLI with --enable-cli and --litellm
# Uses LiteLLM gateway with deepseek-coder model
./rlm-orchestrator/target/release/rlm "$TEMP_FILE" "$QUERY" --enable-cli --litellm -m deepseek-coder -vv

echo ""
echo "================================================================"
echo -e "${BLUE}Demo Complete${NC}"
echo "================================================================"
echo ""
echo "This demo used L1+L3 hybrid approach:"
echo "  - L1 regex extracted ms values from $LINE_COUNT log lines"
echo "  - L3 CLI sorted and computed percentiles"
echo "  - Ground truth: p50=${P50}ms, p95=${P95}ms, p99=${P99}ms"
echo ""

# Cleanup
rm -f "$TEMP_FILE"
