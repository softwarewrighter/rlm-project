#!/bin/bash
# Level 4: War and Peace Family Tree Demo
#
# Demonstrates efficient handling of large files (3.3MB → 57KB)
# by using deterministic extraction BEFORE LLM processing.
#
# Strategy:
#   1. Pre-process: Extract character names + relationship sentences (L3 CLI)
#   2. Analyze: Use LLM only on filtered data (L4 llm_reduce)
#   3. Synthesize: Build family trees from relationships (L4 llm_query)

set -e
cd "$(dirname "$0")/../.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

DATA_FILE="demo/l4/data/war-peace-characters.txt"
QUERY="Build family trees for the main families in War and Peace. Identify the Rostov, Bolkonsky, Kuragin, and Bezukhov families. Show parent-child, spouse, and sibling relationships. Format as structured trees."

clear
echo ""
echo "================================================================"
echo -e "${MAGENTA}    Level 4 Demo: War and Peace Family Tree${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Problem:${NC} Build family trees from 3.3MB novel (65,660 lines)"
echo -e "${CYAN}Naive approach:${NC} 300+ LLM calls, 3385+ seconds, FAILS"
echo ""
echo -e "${CYAN}Efficient approach (this demo):${NC}"
echo "  1. ${GREEN}L3 CLI:${NC} Extract character names + relationship sentences"
echo "     3.3MB → 57KB (98% reduction, <1 second)"
echo "  2. ${GREEN}L4 LLM:${NC} Analyze relationships in filtered data"
echo "     ~3-5 LLM calls on 57KB instead of 300+ on 3.3MB"
echo ""

# Check if data file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${YELLOW}Generating filtered data from War and Peace...${NC}"

    # Check if source file exists
    if [ ! -f "/Users/mike/Downloads/war-and-peace-tolstoy-clean.txt" ]; then
        echo -e "${RED}Error: War and Peace source file not found${NC}"
        echo "Expected at: /Users/mike/Downloads/war-and-peace-tolstoy-clean.txt"
        exit 1
    fi

    # Compile extraction tool if needed
    if [ ! -f "demo/l4/tools/extract-characters" ]; then
        echo "  Compiling extraction tool..."
        rustc -O demo/l4/tools/extract-characters.rs -o demo/l4/tools/extract-characters
    fi

    # Run extraction
    echo "  Extracting characters and relationships..."
    ./demo/l4/tools/extract-characters < /Users/mike/Downloads/war-and-peace-tolstoy-clean.txt > "$DATA_FILE"
fi

# Show file stats
ORIG_SIZE=$(wc -c < /Users/mike/Downloads/war-and-peace-tolstoy-clean.txt 2>/dev/null || echo "3339794")
FILT_SIZE=$(wc -c < "$DATA_FILE")
FILT_LINES=$(wc -l < "$DATA_FILE")
REDUCTION=$(echo "scale=1; (1 - $FILT_SIZE / $ORIG_SIZE) * 100" | bc)

echo -e "${YELLOW}Data reduction:${NC}"
echo "  Original:  $(echo $ORIG_SIZE | awk '{printf "%\047d", $1}') bytes (65,660 lines)"
echo "  Filtered:  $(echo $FILT_SIZE | awk '{printf "%\047d", $1}') bytes ($FILT_LINES lines)"
echo "  Reduction: ${REDUCTION}%"
echo ""

echo -e "${YELLOW}Sample of extracted data:${NC}"
head -20 "$DATA_FILE" | sed 's/^/  /'
echo "  ..."
echo ""

echo -e "${CYAN}Query:${NC}"
echo "  $QUERY"
echo ""

echo "================================================================"
echo -e "${GREEN}Running RLM on filtered data...${NC}"
echo "================================================================"
echo ""

# Run RLM with coordinator mode on the filtered data
./rlm-orchestrator/target/release/rlm "$DATA_FILE" "$QUERY" \
    --enable-llm-delegation \
    --coordinator-mode \
    --litellm \
    -m deepseek-coder \
    --max-iterations 10 \
    -vv

echo ""
echo "================================================================"
echo -e "${MAGENTA}Demo Complete${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Key insight:${NC}"
echo "  Don't send 3.3MB to LLMs. Extract relevant data first."
echo "  98% of War and Peace is NOT about family relationships."
echo ""
echo -e "${CYAN}This demo used:${NC}"
echo "  - L3 CLI (Rust): Deterministic extraction (<1 second)"
echo "  - L4 LLM: Semantic analysis only on filtered 57KB"
echo ""
