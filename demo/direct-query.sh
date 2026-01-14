#!/bin/bash
# Direct query to Ollama without RLM - will fail to find the hidden code

echo "Sending truncated document to qwen3:14b..."
echo ""

CONTEXT=$(head -c 4000 demo/context_too_big.txt)

curl -s http://big72.local:11434/api/generate \
  -d "{\"model\":\"qwen3:14b\",\"prompt\":\"What is the secret launch code in this document?\\n\\n${CONTEXT}\\n...[TRUNCATED - 15KB total]\\n\\nAnswer:\",\"stream\":false}" \
  | jq -r '.response' | head -10
