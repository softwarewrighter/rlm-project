#!/bin/bash
# Level 4: Detective Mystery Demo (Recursive LLM)
#
# Tests llm_delegate for semantic analysis of witness statements.
# The coordinator LLM breaks down the problem and delegates to workers.
#
# Expected time: 30-60 seconds
# Expected iterations: 3-5

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../common.sh"

echo "============================================"
echo "Detective Mystery Demo (Level 4: Recursive LLM)"
echo "============================================"
echo ""
echo "This demo shows how the coordinator LLM delegates"
echo "semantic analysis tasks to worker LLMs."
echo ""

# Create temp file for context
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

# Fetch the sample data from the server
fetch_sample "/samples/detective-mystery" "$CONTEXT_FILE"

# Show context size
CONTEXT_SIZE=$(wc -c < "$CONTEXT_FILE" | tr -d ' ')
echo "Context size: $CONTEXT_SIZE bytes"
echo ""

# Run the query
QUERY="Who murdered Lord Ashford? Cross-reference the witness statements with the physical evidence and identify the killer. Provide your conclusion with supporting evidence."

run_demo "$CONTEXT_FILE" "$QUERY" -vv --coordinator-mode
