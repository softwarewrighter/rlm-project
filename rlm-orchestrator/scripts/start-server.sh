#!/bin/bash
# Start RLM Server with environment variables from ~/.env
#
# The server requires LITELLM_API_KEY (or LITELLM_MASTER_KEY) to be set.
# These are loaded from ~/.env which should contain:
#   LITELLM_API_KEY=sk-local-test-key-123
#   LITELLM_MASTER_KEY=sk-local-test-key-123
#   DEEPSEEK_API_KEY=your_deepseek_key
#
# Usage:
#   ./scripts/start-server.sh           # Start in foreground
#   ./scripts/start-server.sh &         # Start in background
#   ./scripts/start-server.sh --check   # Just check if server is running

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Check for --check flag
if [ "$1" = "--check" ]; then
    if pgrep -f "rlm-server" > /dev/null; then
        echo "Server is running (PID: $(pgrep -f 'rlm-server'))"
        curl -s http://localhost:8080/health
        exit 0
    else
        echo "Server is not running"
        exit 1
    fi
fi

# Load environment variables from ~/.env
if [ -f ~/.env ]; then
    echo "Loading environment from ~/.env..."
    export $(grep -E "^[A-Z]" ~/.env | xargs)
else
    echo "Warning: ~/.env not found"
    echo "Create ~/.env with at least:"
    echo "  LITELLM_API_KEY=sk-local-test-key-123"
    echo "  LITELLM_MASTER_KEY=sk-local-test-key-123"
    exit 1
fi

# Check required vars
if [ -z "$LITELLM_API_KEY" ] && [ -z "$LITELLM_MASTER_KEY" ]; then
    echo "Error: LITELLM_API_KEY or LITELLM_MASTER_KEY must be set in ~/.env"
    exit 1
fi

# Check if already running
if pgrep -f "rlm-server" > /dev/null; then
    echo "Server already running (PID: $(pgrep -f 'rlm-server'))"
    echo "Kill it first: pkill -f rlm-server"
    exit 1
fi

# Check if binary exists
BINARY="$PROJECT_ROOT/target/release/rlm-server"
if [ ! -f "$BINARY" ]; then
    echo "Binary not found: $BINARY"
    echo "Run: cargo build --release"
    exit 1
fi

# Start server
echo "Starting RLM server..."
cd "$PROJECT_ROOT"
exec "$BINARY" config.toml
