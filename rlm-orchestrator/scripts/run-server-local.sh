#!/bin/bash
# Run RLM server using LOCAL ONLY models via LiteLLM
#
# No cloud providers - all requests go to LAN Ollama servers:
# - manager (localhost)
# - big72
# - curiosity
# - hive
#
# LiteLLM handles load balancing and failover.
# Usage tracked at http://localhost:4000/ui

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

# Check for API key (still needed for LiteLLM auth)
if [ -z "$LITELLM_API_KEY" ] && [ -z "$LITELLM_MASTER_KEY" ]; then
    echo "Error: LITELLM_API_KEY or LITELLM_MASTER_KEY not set"
    echo "  export LITELLM_API_KEY=sk-..."
    exit 1
fi

# Check if binary exists
if [ ! -f "./target/release/rlm-server" ]; then
    echo "Binary not found. Building..."
    cargo build --release
fi

# Check if LiteLLM is reachable
if ! curl -s http://localhost:4000/health > /dev/null 2>&1; then
    echo "Warning: LiteLLM not reachable at localhost:4000"
    echo "Start LiteLLM first:"
    echo "  cd ~/github/softwarewrighter/emacs-ai-api/llm-gateway"
    echo "  docker compose up -d litellm"
    exit 1
fi

echo "Starting RLM server with LOCAL ONLY models..."
echo "  Root: local-root (qwen2.5-coder:14b across LAN)"
echo "  Sub:  local-sub (gemma2:9b/mistral:7b across LAN)"
echo ""
exec ./target/release/rlm-server config-local.toml
