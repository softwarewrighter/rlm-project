#!/bin/bash
# Level 4: Detective Mystery Demo (Recursive LLM Delegation)
# Problem Type: Multi-hop semantic reasoning
#
# This demo shows how L4 llm_delegate works for semantic analysis:
# 1. L1 DSL commands extract witness statements and evidence
# 2. L4 llm_delegate analyzes each section with nested RLM (has tool access)
# 3. Root RLM synthesizes findings and identifies the murderer
#
# The mystery requires cross-referencing multiple witnesses and evidence
# to identify contradictions and build a case - tasks that DSL/WASM can't do.

set -e
cd "$(dirname "$0")/../.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

DATA_FILE="demo/l4/data/detective-mystery.txt"
# Query explicitly asks for ALL evidence to get consistent, complete responses
QUERY="Who murdered Lord Ashford? Analyze ALL witness statements and ALL physical evidence. Identify the killer and provide your conclusion with COMPLETE supporting evidence including: motive, opportunity, physical evidence (footprints, weapon, poison), and any timeline contradictions."
SERVER_URL="${SERVER_URL:-http://localhost:4539}"

clear
echo ""
echo "================================================================"
echo -e "${MAGENTA}    Level 4 Demo: The Ashford Manor Murder${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Problem Type:${NC} Multi-hop semantic reasoning"
echo -e "${CYAN}Dataset:${NC} 30KB case file with 7 witnesses + evidence"
echo -e "${CYAN}Strategy:${NC} L1 extract sections, L4 delegate for semantic analysis"
echo ""
echo -e "${CYAN}Key Challenge:${NC}"
echo "  DSL can extract text but can't understand contradictions."
echo "  WASM/CLI can compute but can't reason about alibis."
echo "  llm_delegate creates nested RLM with tool access for each section."
echo ""
echo -e "${CYAN}Query:${NC}"
echo "  $QUERY"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file not found: $DATA_FILE${NC}"
    exit 1
fi

# Show file stats
LINE_COUNT=$(wc -l < "$DATA_FILE")
CHAR_COUNT=$(wc -c < "$DATA_FILE")
echo -e "${YELLOW}Case file:${NC} $LINE_COUNT lines, $CHAR_COUNT bytes"
echo ""

# Show structure
echo -e "${YELLOW}Case file structure:${NC}"
grep "^===\|^---\|^\[WITNESS\|^\[EVIDENCE" "$DATA_FILE" | head -20
echo "..."
echo ""

# Check if RLM server is running with LLM delegation enabled
if ! curl -s "$SERVER_URL/health" > /dev/null 2>&1; then
    echo -e "${YELLOW}RLM server not running. Start it with:${NC}"
    echo ""
    echo "  cd rlm-orchestrator"
    echo "  export \$(cat ~/.env | grep -v '^#' | xargs)"
    echo "  ./target/release/rlm-server config-litellm-cli.toml --enable-llm-delegation"
    echo ""
    exit 1
fi

echo "================================================================"
echo -e "${GREEN}Running RLM with L4 LLM Delegation...${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Expected workflow:${NC}"
echo "  1. L1: Extract witness statements using regex"
echo "  2. L4: llm_delegate to analyze each witness for key claims"
echo "  3. L1: Extract physical evidence"
echo "  4. L4: llm_delegate to cross-reference evidence with witness timelines"
echo "  5. Synthesize and identify the murderer with reasoning"
echo ""

# Run RLM CLI with coordinator mode (uses llm_reduce for efficient large context processing)
./rlm-orchestrator/target/release/rlm "$DATA_FILE" "$QUERY" \
    --enable-llm-delegation \
    --coordinator-mode \
    --litellm \
    -m deepseek-coder \
    --max-iterations 15 \
    -vv

echo ""
echo "================================================================"
echo -e "${MAGENTA}Demo Complete${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Ground Truth:${NC}"
echo "  The murderer is Colonel Arthur Pemberton"
echo ""
echo -e "${CYAN}Key Evidence:${NC}"
echo "  - Motive: Fraud exposure from 1998 incident"
echo "  - Opportunity: Admitted presence in study at time of death"
echo "  - Physical: Footprints matching his distinctive limp"
echo "  - Witness: Gardener saw limping figure at 10:20 PM and 10:30-35 PM"
echo "  - Timeline gap: Claims he left at 10:20 but gardener saw him at 10:30+"
echo ""
echo "This demo used L4 llm_delegate for:"
echo "  - Semantic analysis of witness statements"
echo "  - Cross-referencing alibis with physical evidence"
echo "  - Identifying contradictions between testimonies"
echo ""
