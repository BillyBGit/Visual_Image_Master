#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.img_sort.pid"

if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found — img_sort may not be running."
    exit 0
fi

PID=$(cat "$PIDFILE")

if kill -0 "$PID" 2>/dev/null; then
    kill -TERM "$PID" 2>/dev/null || true
    # Wait up to 5 seconds for clean exit
    for i in $(seq 1 10); do
        sleep 0.5
        kill -0 "$PID" 2>/dev/null || break
    done
    # Force kill if still running
    if kill -0 "$PID" 2>/dev/null; then
        echo "Process $PID did not exit cleanly, sending SIGKILL."
        kill -9 "$PID" 2>/dev/null || true
    fi
    echo "img_sort stopped (PID $PID)."
else
    echo "Process $PID was not running."
fi

rm -f "$PIDFILE"
