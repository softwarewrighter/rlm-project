#!/bin/bash
# Stop RLM server running on port 8080

PORT=8080
PID=$(lsof -ti :$PORT 2>/dev/null)

if [ -n "$PID" ]; then
    echo "Stopping RLM server (PID: $PID)..."
    kill $PID
    sleep 1
    # Force kill if still running
    if kill -0 $PID 2>/dev/null; then
        echo "Force killing..."
        kill -9 $PID
    fi
    echo "Stopped."
else
    echo "No server running on port $PORT"
fi
