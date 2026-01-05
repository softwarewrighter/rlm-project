#!/bin/bash
# Build all RLM orchestrator binaries
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Building RLM orchestrator..."
cargo build --release

echo ""
echo "Build complete. Binaries available at:"
echo "  $PROJECT_DIR/target/release/rlm-server"
echo "  $PROJECT_DIR/target/release/rlm-test"
