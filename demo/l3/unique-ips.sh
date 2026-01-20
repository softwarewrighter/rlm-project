#!/bin/bash
# Level 3 CLI: Unique IP Analysis Demo (5000 lines)
# Problem Type: BrowseComp-Plus
#
# This demo shows how L1+L3 hybrid approach works:
# 1. L1 DSL commands (regex) extract IP addresses from log lines
# 2. L3 rust_cli_intent counts unique IPs and ranks by frequency
#
# This hybrid approach:
# - L1 regex extracts just the IP addresses (simpler input for L3)
# - L3 CLI uses HashSet for unique counting, HashMap for frequency
# - Avoids complex parsing in the generated Rust code

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
QUERY="How many unique IP addresses appear in these logs? List the top 10 most active IPs."
SERVER_URL="${SERVER_URL:-http://localhost:4539}"

clear
echo ""
echo "================================================================"
echo -e "${BLUE}    Level 3 CLI Demo: Unique IP Analysis (5000 lines)${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Problem Type:${NC} BrowseComp-Plus"
echo -e "${CYAN}Dataset:${NC} 5000 log lines with various IP addresses"
echo -e "${CYAN}Strategy:${NC} L1 regex extracts IPs, L3 CLI counts/ranks"
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
UNIQUE_IPS=$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$TEMP_FILE" | sort -u | wc -l)
echo -e "${GREEN}Downloaded:${NC} $LINE_COUNT lines, $CHAR_COUNT bytes"
echo -e "${GREEN}Unique IPs:${NC} $UNIQUE_IPS (ground truth)"
echo ""

# Show sample of data
echo -e "${YELLOW}Sample IPs found in logs:${NC}"
grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$TEMP_FILE" | sort -u | head -5
echo "..."
echo ""

echo "================================================================"
echo -e "${GREEN}Running RLM with CLI enabled...${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Expected workflow:${NC}"
echo "  1. L1: regex to extract IP addresses from each line"
echo "  2. L3: rust_cli_intent with HashSet for unique count"
echo "  3. L3: HashMap to count frequency, sort for top 10"
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
echo "  - L1 regex extracted IPs from 5000 log lines"
echo "  - L3 CLI used HashSet for unique count, HashMap for frequency"
echo "  - Expected unique IPs: $UNIQUE_IPS"
echo ""

# Cleanup
rm -f "$TEMP_FILE"
