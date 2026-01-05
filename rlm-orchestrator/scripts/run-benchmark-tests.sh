#!/bin/bash
# Run RLM benchmark tests
# Requires the RLM server to be running (use run-server.sh)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if server is running
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "Error: RLM server not running on http://localhost:8080"
    echo "Start it first with: ./scripts/run-server.sh"
    exit 1
fi

# Build if needed
if [ ! -f "target/release/rlm-test" ]; then
    echo "Building first..."
    cargo build --release --bin rlm-test
fi

echo "Running benchmark tests..."
echo "Server: http://localhost:8080"
echo ""

exec ./target/release/rlm-test "$@"
