#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_CMD="uvicorn app:app --host 0.0.0.0 --port 8000 --reload"
FRONTEND_CMD="npm run dev -- --host 0.0.0.0 --port 5173"

# Ensure dependencies are installed
"${SCRIPT_DIR}/setup_all.sh"

# Start backend
cd "$REPO_ROOT"
source .venv/bin/activate
(
  echo "[one-click] Starting backend server..."
  ${BACKEND_CMD}
) &
BACK_PID=$!

deactivate >/dev/null 2>&1 || true

# Start frontend
cd "$REPO_ROOT/frontend"
(
  echo "[one-click] Starting frontend dev server..."
  ${FRONTEND_CMD}
) &
FRONT_PID=$!

cleanup() {
  echo "\n[one-click] Shutting down services..."
  kill "$BACK_PID" >/dev/null 2>&1 || true
  kill "$FRONT_PID" >/dev/null 2>&1 || true
  wait "$BACK_PID" >/dev/null 2>&1 || true
  wait "$FRONT_PID" >/dev/null 2>&1 || true
}

trap cleanup INT TERM

echo "[one-click] Backend PID: $BACK_PID"
echo "[one-click] Frontend PID: $FRONT_PID"
echo "[one-click] Press Ctrl+C to stop both servers."

wait -n "$BACK_PID" "$FRONT_PID"
cleanup
