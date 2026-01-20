#!/bin/bash
# Level 4: War and Peace Family Tree Demo
#
# Demonstrates RLM's ability to handle larger-than-context files by having
# the LLM intelligently choose tools to reduce the data before analysis.
#
# The LLM should recognize:
#   1. The 3.3MB file is too large to analyze directly
#   2. Use L3 CLI (rust_cli_intent) to extract relevant character/relationship data
#   3. Use L4 LLM (llm_reduce) to analyze the filtered data semantically
#
# This is NOT pre-processed - RLM orchestrates the entire pipeline.

set -e
cd "$(dirname "$0")/../.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Use the FULL War and Peace file - RLM must figure out how to handle it
DATA_FILE="/Users/mike/Downloads/war-and-peace-tolstoy-clean.txt"
QUERY="Build ASCII family tree diagrams for the main noble families: Rostov, Bolkonsky, Bezukhov, and Kuragin. Use tree notation with └── and ├── to show parent-child relationships. Mark spouses with (m. Name). Show ALL family members found in the text."

clear
echo ""
echo "================================================================"
echo -e "${MAGENTA}    Level 4 Demo: War and Peace Family Tree${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}Challenge:${NC} Build family trees from 3.3MB novel (65,660 lines)"
echo ""
echo -e "${CYAN}Why this is hard:${NC}"
echo "  - File is WAY too large to send to an LLM directly"
echo "  - Naive chunking would require 300+ LLM calls and still fail"
echo "  - 98% of the text is NOT about family relationships"
echo ""
echo -e "${CYAN}RLM's approach (LLM-orchestrated):${NC}"
echo "  1. LLM recognizes the file is too large"
echo "  2. LLM uses ${GREEN}rust_cli_intent${NC} to generate extraction code"
echo "     (extracts character names + relationship sentences)"
echo "  3. LLM uses ${GREEN}llm_reduce${NC} on the filtered data"
echo "  4. LLM synthesizes family trees from relationships"
echo ""
echo -e "${YELLOW}Key insight:${NC} The LLM decides HOW to process the data."
echo "  RLM provides tools; the LLM orchestrates their use."
echo ""

# Check if source file exists
if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: War and Peace source file not found${NC}"
    echo "Expected at: $DATA_FILE"
    echo ""
    echo "Download from Project Gutenberg or use a clean text version."
    exit 1
fi

# Show file stats
FILE_SIZE=$(wc -c < "$DATA_FILE")
FILE_LINES=$(wc -l < "$DATA_FILE")

echo -e "${YELLOW}Input file:${NC}"
echo "  Path:  $DATA_FILE"
echo "  Size:  $(echo $FILE_SIZE | awk '{printf "%\047d", $1}') bytes"
echo "  Lines: $(echo $FILE_LINES | awk '{printf "%\047d", $1}')"
echo ""

echo -e "${CYAN}Query:${NC}"
echo "  $QUERY"
echo ""

echo "================================================================"
echo -e "${GREEN}Running RLM on full 3.3MB file...${NC}"
echo -e "${YELLOW}Watch as the LLM decides how to handle this large input.${NC}"
echo "================================================================"
echo ""

# Run RLM with LLM delegation and CLI on the FULL file
# CLI enables phased processing for large contexts
# The LLM must figure out to use rust_cli_intent for extraction
./rlm-orchestrator/target/release/rlm "$DATA_FILE" "$QUERY" \
    --enable-llm-delegation \
    --enable-cli \
    --litellm \
    -m deepseek-coder \
    --max-iterations 20 \
    -vv

echo ""
echo "================================================================"
echo -e "${MAGENTA}Demo Complete${NC}"
echo "================================================================"
echo ""
echo -e "${CYAN}What happened:${NC}"
echo "  1. RLM received 3.3MB input (too large for direct LLM processing)"
echo "  2. The LLM recognized this and used rust_cli_intent to extract"
echo "     only character names and relationship sentences"
echo "  3. The LLM then used llm_reduce on the filtered data"
echo "  4. Family trees were synthesized from the relationships"
echo ""
echo -e "${CYAN}This demonstrates RLM's core value:${NC}"
echo "  LLMs can process arbitrarily large files by intelligently"
echo "  choosing which tools to use for data reduction."
echo ""
