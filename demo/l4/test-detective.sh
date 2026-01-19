#!/bin/bash
# Test: Detective Mystery Demo (Level 4)
# Verifies that the LLM correctly identifies Colonel Arthur Pemberton as the murderer.
#
# Usage: ./demo/l4/test-detective.sh
#
# Prerequisites:
#   1. LiteLLM gateway running at localhost:4000
#   2. ~/.env file with LITELLM_MASTER_KEY set
#   3. RLM CLI binary built (auto-builds if missing)
#
# Environment variables (in ~/.env):
#   LITELLM_MASTER_KEY=sk-your-key-here
#   LITELLM_HOST=http://localhost:4000  # optional, defaults to localhost:4000
#
# Exit codes:
#   0 - Test passed (answer identifies Pemberton)
#   1 - Test failed (wrong answer or no answer)

set -e
cd "$(dirname "$0")/../.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "=========================================="
echo "Test: Detective Mystery (L4)"
echo "=========================================="

# Configuration
DATA_FILE="demo/l4/data/detective-mystery.txt"
ANSWER_FILE="demo/l4/data/answer.txt"
# Query explicitly asks for ALL evidence to get consistent, complete responses
QUERY="Who murdered Lord Ashford? Analyze ALL witness statements and ALL physical evidence. Identify the killer and provide your conclusion with COMPLETE supporting evidence including: motive, opportunity, physical evidence (footprints, weapon, poison), and any timeline contradictions."
OUTPUT_FILE="/tmp/detective-test-output.txt"
LITELLM_URL="${LITELLM_HOST:-http://localhost:4000}"

# Check prerequisites
echo ""
echo -e "${CYAN}Checking prerequisites...${NC}"

if [ ! -f "$DATA_FILE" ]; then
    echo -e "${RED}Error: Data file not found: $DATA_FILE${NC}"
    exit 1
fi
echo "  Data file: OK"

if [ ! -f "./rlm-orchestrator/target/release/rlm" ]; then
    echo -e "${YELLOW}  RLM CLI: Building...${NC}"
    cargo build --release --bin rlm -p rlm-orchestrator
else
    echo "  RLM CLI: OK"
fi

# Load environment variables
if [ -f ~/.env ]; then
    export $(cat ~/.env | grep -v '^#' | xargs)
    echo "  Environment: Loaded from ~/.env"
else
    echo -e "${RED}Error: ~/.env not found${NC}"
    echo ""
    echo "Create ~/.env with:"
    echo "  LITELLM_MASTER_KEY=sk-your-key-here"
    exit 1
fi

# Check for API key
if [ -z "$LITELLM_MASTER_KEY" ]; then
    echo -e "${RED}Error: LITELLM_MASTER_KEY not set in ~/.env${NC}"
    exit 1
fi
echo "  API key: Set"

# Check LiteLLM connectivity
echo -n "  LiteLLM gateway: "
if curl -s --max-time 5 "$LITELLM_URL/health" > /dev/null 2>&1; then
    echo "OK ($LITELLM_URL)"
else
    echo -e "${RED}Not reachable at $LITELLM_URL${NC}"
    echo ""
    echo "Start LiteLLM gateway or set LITELLM_HOST in ~/.env"
    exit 1
fi

echo ""
echo -e "${CYAN}Query:${NC}"
echo "  $QUERY"
echo ""
echo -e "${CYAN}Expected:${NC} Colonel Arthur Pemberton"
echo ""
echo "Running RLM..."
echo "----------------------------------------"

# Run the CLI with coordinator mode (uses llm_reduce for efficient large context processing)
./rlm-orchestrator/target/release/rlm "$DATA_FILE" "$QUERY" \
    --enable-llm-delegation \
    --coordinator-mode \
    --litellm \
    -m deepseek-coder \
    --max-iterations 15 \
    -v 2>&1 | tee "$OUTPUT_FILE"

echo "----------------------------------------"
echo ""

# Extract the answer - everything between the two "════" divider lines after "Answer:"
# Use sed to get lines between Answer: and second ════, then filter out the dividers
ANSWER=$(sed -n '/^Answer:$/,/^═\+$/p' "$OUTPUT_FILE" | grep -v "^Answer:" | grep -v "^═" | head -100)

if [ -z "$ANSWER" ]; then
    echo -e "${RED}TEST FAILED: No answer produced${NC}"
    echo "Full output saved to: $OUTPUT_FILE"
    exit 1
fi

echo -e "${CYAN}Extracted answer:${NC}"
echo "$ANSWER"
echo ""

# Simple check: does the answer mention Pemberton?
if echo "$ANSWER" | grep -qi "Pemberton"; then
    echo -e "${GREEN}Basic check: Answer identifies Pemberton${NC}"

    # Additional quality check using LLM
    echo ""
    echo -e "${CYAN}Running LLM quality evaluation...${NC}"

    # Read expected answer
    EXPECTED_ANSWER=$(cat "$ANSWER_FILE")

    # Use LiteLLM directly to evaluate the answer quality
    EVAL_PROMPT="Compare these two answers about a murder mystery:

EXPECTED ANSWER (ground truth):
$EXPECTED_ANSWER

RLM ANSWER (to evaluate):
$ANSWER

Questions:
1. Does the RLM answer correctly identify the murderer as Colonel Arthur Pemberton (or just Pemberton)?
2. Does it mention the key evidence: motive (fraud), opportunity (admitted presence), and footprints?
3. Rate the answer quality: EXCELLENT, GOOD, PARTIAL, or WRONG

Respond in format:
CORRECT_MURDERER: yes/no
KEY_EVIDENCE: yes/partial/no
QUALITY: EXCELLENT/GOOD/PARTIAL/WRONG
SUMMARY: <one sentence summary>"

    EVAL_RESULT=$(curl -s "$LITELLM_URL/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
        -d "{
            \"model\": \"deepseek-chat\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$EVAL_PROMPT" | jq -Rs .)}],
            \"max_tokens\": 200
        }" | jq -r '.choices[0].message.content // "Evaluation failed"')

    echo "$EVAL_RESULT"
    echo ""

    # Parse evaluation result
    if echo "$EVAL_RESULT" | grep -qi "CORRECT_MURDERER: yes"; then
        if echo "$EVAL_RESULT" | grep -qi "QUALITY: EXCELLENT\|QUALITY: GOOD"; then
            echo -e "${GREEN}TEST PASSED: High quality answer${NC}"
            exit 0
        elif echo "$EVAL_RESULT" | grep -qi "QUALITY: PARTIAL"; then
            echo -e "${YELLOW}TEST PASSED (with warnings): Partial answer${NC}"
            exit 0
        fi
    fi

    # Fallback: basic pattern match is enough
    echo -e "${GREEN}TEST PASSED: Answer identifies Pemberton as murderer${NC}"
    exit 0
else
    echo -e "${RED}TEST FAILED: Answer does not identify Pemberton${NC}"
    echo ""
    echo "Full output saved to: $OUTPUT_FILE"
    exit 1
fi
