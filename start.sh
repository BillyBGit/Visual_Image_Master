#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$SCRIPT_DIR/.img_sort.pid"
VENV="$HOME/scripts/venv"

if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "img_sort is already running (PID $PID). Run stop.sh first."
        exit 1
    else
        echo "Stale PID file found (PID $PID not running). Removing."
        rm -f "$PIDFILE"
    fi
fi

source "$VENV/bin/activate"

cd "$SCRIPT_DIR"
python main.py </dev/null &
APP_PID=$!
echo "$APP_PID" > "$PIDFILE"

echo "img_sort started (PID $APP_PID)"
