#!/bin/bash
# Run RLM server using LiteLLM proxy for usage tracking
#
# Requires:
# - LiteLLM running on localhost:4000
# - LITELLM_API_KEY or LITELLM_MASTER_KEY environment variable
#
# Usage:
#   export LITELLM_API_KEY=sk-...
#   ./scripts/run-server-litellm.sh

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
    echo "Start LiteLLM first: litellm --config config.yaml"
fi

echo "Starting RLM server with LiteLLM proxy..."
exec ./target/release/rlm-server config-litellm.toml
