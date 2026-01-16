#!/bin/bash
# War and Peace Family Tree Demo
#
# Analyzes 3.2MB of War and Peace text to extract character relationships.
# This demonstrates RLM's ability to handle very large contexts efficiently.
#
# Expected time with DeepSeek: 60-90 seconds
# Expected iterations: 3-5

source "$(dirname "$0")/common.sh"

echo "============================================"
echo "War and Peace Family Tree Demo"
echo "============================================"
echo ""
echo "This demo analyzes the full text of War and Peace (~3.2MB)"
echo "to identify and map character relationships."
echo ""

# Create temp file for context
CONTEXT_FILE=$(mktemp)
trap "rm -f $CONTEXT_FILE" EXIT

# Fetch the sample data
fetch_sample "/samples/war-and-peace" "$CONTEXT_FILE"

# Run the query
QUERY="Build a family tree for the main characters. Identify characters who appear multiple times and are related to each other (by blood or marriage). Show the relationships in a structured format."

run_demo "$CONTEXT_FILE" "$QUERY" -vv
