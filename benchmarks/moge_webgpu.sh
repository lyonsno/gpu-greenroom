#!/usr/bin/env bash
# WebGPU benchmark wrapper for MoGe-2.
#
# Starts the moge-webgpu dev server, runs the Puppeteer benchmark,
# kills the server, and outputs JSON to stdout.
#
# Usage:
#   benchmarks/moge_webgpu.sh [--runs N] [--output-dir /path]
#
# Designed to run standalone or through the GPU Greenroom queue.

set -euo pipefail

MOGE_WEBGPU_DIR="${MOGE_WEBGPU_DIR:-$HOME/dev/moge-webgpu}"
PORT="${BENCHMARK_PORT:-5181}"
RUNS=10
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

echo "WebGPU MoGe-2 benchmark -- port=$PORT, runs=$RUNS" >&2
echo "moge-webgpu dir: $MOGE_WEBGPU_DIR" >&2

# Do not kill an unrelated process. Greenroom serializes GPU work, not TCP
# ownership; an occupied benchmark port is a route/setup failure.
if lsof -ti :"$PORT" >/dev/null 2>&1; then
    echo "ERROR: Port $PORT is already in use; choose BENCHMARK_PORT or --port" >&2
    exit 1
fi

# Start dev server in background
cd "$MOGE_WEBGPU_DIR"
npx vite --port "$PORT" --strictPort &
SERVER_PID=$!

# Cleanup function
cleanup() {
    echo "Stopping dev server (pid $SERVER_PID)..." >&2
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for dev server on port $PORT..." >&2
for i in $(seq 1 30); do
    if curl -s -o /dev/null "http://localhost:$PORT/" 2>/dev/null; then
        echo "Server ready after ${i}s" >&2
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Dev server died during startup" >&2
        exit 1
    fi
    sleep 1
done

# Verify server is actually responding
if ! curl -s -o /dev/null "http://localhost:$PORT/"; then
    echo "ERROR: Dev server not responding after 30s" >&2
    exit 1
fi

# Run the Puppeteer benchmark
echo "Running Puppeteer benchmark ($RUNS runs)..." >&2
RESULT=$(node "$MOGE_WEBGPU_DIR/tools/benchmark.mjs" --port "$PORT" --runs "$RUNS" --json)

# Output JSON to stdout
echo "$RESULT"

# Write to output dir if specified
if [[ -n "$OUTPUT_DIR" ]]; then
    mkdir -p "$OUTPUT_DIR"
    echo "$RESULT" > "$OUTPUT_DIR/benchmark_webgpu.json"
    echo "Results written to $OUTPUT_DIR/benchmark_webgpu.json" >&2
fi
