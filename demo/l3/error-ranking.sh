#!/bin/bash
# Level 3 CLI: Error Type Frequency Demo (5000 lines)
# Problem Type: BrowseComp-Plus
#
# This demo shows how L1+L3 hybrid approach works:
# 1. L1 DSL commands (find/regex) pre-filter to only ERROR lines
# 2. L3 rust_cli_intent counts and ranks the filtered error types
#
# This hybrid approach:
# - Reduces input size for the Rust code (5000 -> ~700 lines)
# - Simplifies the generated Rust code (no need to filter)
# - Keeps HashMap-based frequency counting in native code

set -e
cd "$(dirname "$0")/../.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

TEMP_FILE="/tmp/large-logs.txt"
QUERY="Rank the error types from most to least frequent. Show the count for each error type."
SERVER_URL="${SERVER_URL:-http://localhost:4539}"

clear
echo ""
echo "================================================================"
echo -e "${BLUE}    Level 3 CLI Demo: Error Type Frequency (5000 lines)${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Problem Type:${NC} BrowseComp-Plus"
echo -e "${CYAN}Dataset:${NC} 5000 log lines with various error types"
echo -e "${CYAN}Strategy:${NC} L1 find/regex to filter, L3 CLI to count/rank"
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

echo -e "${YELLOW}Fetching 5000-line log dataset...${NC}"
curl -s "$SERVER_URL/samples/large-logs" > "$TEMP_FILE"
LINE_COUNT=$(wc -l < "$TEMP_FILE")
CHAR_COUNT=$(wc -c < "$TEMP_FILE")
ERROR_COUNT=$(grep -c "ERROR" "$TEMP_FILE" || echo "0")
echo -e "${GREEN}Downloaded:${NC} $LINE_COUNT lines, $CHAR_COUNT bytes"
echo -e "${GREEN}ERROR lines:${NC} $ERROR_COUNT (this is what L1 will filter to)"
echo ""

# Show sample of data
echo -e "${YELLOW}Sample ERROR lines:${NC}"
grep "ERROR" "$TEMP_FILE" | head -3
echo "..."
echo ""

echo "================================================================"
echo -e "${GREEN}Running RLM with CLI enabled...${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Expected workflow:${NC}"
echo "  1. L1: find 'ERROR' or regex to filter log lines"
echo "  2. L3: rust_cli_intent to count each error type with HashMap"
echo "  3. Return ranked frequency list"
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
echo "  - L1 DSL filtered 5000 lines -> ~$ERROR_COUNT ERROR lines"
echo "  - L3 CLI counted and ranked error types with HashMap"
echo ""

# Cleanup
rm -f "$TEMP_FILE"
