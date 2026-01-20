#!/bin/bash
# RLM Demo: Show small-context LLM failing, then succeeding with RLM
#
# Uses War and Peace (3.3MB, ~800K tokens) with a hidden needle.
# The needle "NEPTUNE-FALCON-7749" is buried at line 33,002.

set -e
cd "$(dirname "$0")/.."

DEMO_DIR="demo"
CONTEXT_FILE="$DEMO_DIR/war-and-peace-with-needle.txt"
QUERY="What is the hidden passphrase mentioned in the author's secret note?"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear
echo ""
echo "================================================================"
echo -e "${BLUE}    RLM DEMO: Needle in a Haystack with War and Peace${NC}"
echo "================================================================"
echo ""

# Show document stats
CHAR_COUNT=$(wc -c < "$CONTEXT_FILE")
LINE_COUNT=$(wc -l < "$CONTEXT_FILE")
TOKEN_EST=$((CHAR_COUNT / 4))
echo -e "${YELLOW}Document:${NC} War and Peace by Leo Tolstoy"
echo -e "${YELLOW}Size:${NC} $CHAR_COUNT characters (~$TOKEN_EST tokens)"
echo -e "${YELLOW}Lines:${NC} $LINE_COUNT"
echo ""
echo -e "${CYAN}Hidden needle at line 33,002:${NC}"
echo "  [AUTHOR'S SECRET NOTE: THE HIDDEN PASSPHRASE IS: NEPTUNE-FALCON-7749]"
echo ""
echo -e "${YELLOW}Query:${NC} $QUERY"
echo ""

# Check if ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Ollama is not running. Please start it with 'ollama serve'${NC}"
    exit 1
fi

echo "================================================================"
echo -e "${RED}TEST 1: Direct Query to phi3:3.8b (4K context)${NC}"
echo "================================================================"
echo ""
echo "Attempting to send 3.3MB document to a model with 4K token context..."
echo "This will truncate the document, losing the needle in the middle."
echo ""
echo -e "${YELLOW}Sending request... (this may take a moment)${NC}"
echo ""

# Read first 16K chars (roughly 4K tokens - phi3's limit)
TRUNCATED_CONTEXT=$(head -c 16000 "$CONTEXT_FILE")

DIRECT_RESPONSE=$(curl -s --max-time 120 http://localhost:11434/api/generate \
  -d "{
    \"model\": \"phi3:3.8b\",
    \"prompt\": \"Answer this question based ONLY on the text provided. If the answer is not in the text, say 'Not found in text'.\n\nTEXT:\n$TRUNCATED_CONTEXT\n\nQUESTION: $QUERY\n\nANSWER:\",
    \"stream\": false,
    \"options\": {
      \"num_ctx\": 4096
    }
  }" 2>/dev/null | jq -r '.response // .error // "No response"' 2>/dev/null || echo "Request failed or timed out")

echo -e "${YELLOW}Direct Response:${NC}"
echo "$DIRECT_RESPONSE" | head -10
echo ""

if echo "$DIRECT_RESPONSE" | grep -qi "NEPTUNE-FALCON-7749"; then
    echo -e "${GREEN}[UNEXPECTED] Found the passphrase!${NC}"
else
    echo -e "${RED}[EXPECTED FAILURE] Could not find the passphrase${NC}"
    echo ""
    echo "Why? The document is 3.3MB but phi3 can only see ~16KB."
    echo "The needle is at line 33,002 - completely outside the context window."
fi

echo ""
read -p "Press Enter to continue to RLM test..."
echo ""

echo "================================================================"
echo -e "${GREEN}TEST 2: RLM Query (iterative search)${NC}"
echo "================================================================"
echo ""
echo "Now using RLM to let phi3 EXPLORE the document iteratively..."
echo "The LLM will issue search commands instead of reading everything."
echo ""

# Check if RLM server is running
if ! curl -s http://localhost:4539/health > /dev/null 2>&1; then
    echo -e "${YELLOW}RLM server not running. Start it with:${NC}"
    echo ""
    echo "  cd rlm-orchestrator"
    echo "  CONFIG_PATH=../demo/config-small-llm.toml cargo run --bin rlm-server"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo -e "${YELLOW}Sending RLM query...${NC}"
echo ""

# Query via RLM
CONTEXT_JSON=$(cat "$CONTEXT_FILE" | jq -Rs .)
RLM_RESPONSE=$(curl -s --max-time 300 -X POST http://localhost:4539/debug \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"$QUERY\",
    \"context\": $CONTEXT_JSON
  }" 2>/dev/null)

if [ -z "$RLM_RESPONSE" ]; then
    echo -e "${RED}RLM request failed or timed out${NC}"
    exit 1
fi

ANSWER=$(echo "$RLM_RESPONSE" | jq -r '.answer // .error // "No answer"')
ITERATIONS=$(echo "$RLM_RESPONSE" | jq -r '.iterations // 0')
SUB_CALLS=$(echo "$RLM_RESPONSE" | jq -r '.sub_calls // 0')
BYPASSED=$(echo "$RLM_RESPONSE" | jq -r '.bypassed // false')

echo -e "${CYAN}RLM Execution Summary:${NC}"
echo "  Iterations: $ITERATIONS"
echo "  Sub-LM calls: $SUB_CALLS"
echo "  Bypassed: $BYPASSED"
echo ""
echo -e "${YELLOW}Answer:${NC}"
echo "$ANSWER"
echo ""

if echo "$ANSWER" | grep -qi "NEPTUNE-FALCON-7749"; then
    echo -e "${GREEN}[SUCCESS] RLM found the passphrase!${NC}"
else
    echo -e "${RED}[FAILED] RLM did not find the passphrase${NC}"
    echo ""
    echo "Debug: Check the iteration history with:"
    echo "  curl -s http://localhost:4539/debug -X POST -H 'Content-Type: application/json' -d '{...}' | jq '.iterations_history'"
fi

echo ""
echo "================================================================"
echo -e "${BLUE}SUMMARY${NC}"
echo "================================================================"
echo ""
echo "Document: 3.3MB War and Peace (~800K tokens)"
echo "Model: phi3:3.8b (4K token context = ~16KB)"
echo "Needle: Hidden passphrase at line 33,002"
echo ""
echo "┌─────────────────┬────────────────────────────────────────┐"
echo "│ Direct Query    │ FAILS - context truncated, needle lost │"
echo "├─────────────────┼────────────────────────────────────────┤"
echo "│ RLM Query       │ WORKS - iteratively searches document  │"
echo "└─────────────────┴────────────────────────────────────────┘"
echo ""
echo "RLM enables small models to handle documents 50x their context!"
echo ""
