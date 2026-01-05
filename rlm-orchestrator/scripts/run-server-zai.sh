#!/bin/bash
# Run RLM server using Z.ai GLM-4.7 as root LLM
#
# Z.ai coding plan = quota-based (daily limits, not per-token)
# This makes GLM-4.7 effectively free for heavy testing!
#
# Root: GLM-4.7 (200K context, quota-based)
# Sub:  Local models via LiteLLM (free, tracked)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Source ~/.env if it exists
if [ -f ~/.env ]; then
    set -a
    source ~/.env
    set +a
fi

# Check for API key
if [ -z "$LITELLM_API_KEY" ] && [ -z "$LITELLM_MASTER_KEY" ]; then
    echo "Error: LITELLM_API_KEY or LITELLM_MASTER_KEY not set"
    exit 1
fi

# Check if binary exists
if [ ! -f "./target/release/rlm-server" ]; then
    echo "Binary not found. Building..."
    cargo build --release
fi

# Check if LiteLLM is reachable
if ! curl -s http://localhost:4000/health > /dev/null 2>&1; then
    echo "Error: LiteLLM not reachable at localhost:4000"
    echo "Start LiteLLM first:"
    echo "  cd ~/github/softwarewrighter/emacs-ai-api/llm-gateway"
    echo "  docker compose up -d litellm"
    exit 1
fi

echo "Starting RLM server with Z.ai GLM-4.7..."
echo "  Root: glm-4.7 (Z.ai, quota-based)"
echo "  Sub:  local-sub (LAN Ollama servers)"
echo ""
exec ./target/release/rlm-server config-zai.toml
