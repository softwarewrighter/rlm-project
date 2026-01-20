#!/bin/bash
# Start the RLM server
# Requires DEEPSEEK_API_KEY environment variable for DeepSeek provider
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check for API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Warning: DEEPSEEK_API_KEY not set. DeepSeek provider will fail."
    echo "Set it with: export DEEPSEEK_API_KEY=your-key"
fi

# Build if needed
if [ ! -f "target/release/rlm-server" ]; then
    echo "Building first..."
    cargo build --release --bin rlm-server
fi

echo "Starting RLM server on http://localhost:4539"
echo "Config: $PROJECT_DIR/config.toml"
echo "Press Ctrl+C to stop"
echo ""

exec ./target/release/rlm-server
