#!/usr/bin/env bash
# Starts backend (:8000, auto-reload) and frontend (:5173, HMR) together.
# Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

cleanup() {
  echo ""
  echo "Stopping..."
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting backend on :8000"
(
  cd backend
  source .venv/bin/activate 2>/dev/null || true
  uvicorn app.main:app --reload --port 8000
) &
BACKEND_PID=$!

sleep 1
echo "==> Starting frontend on :5173 (Ctrl+C stops both)"
cd frontend
npm run dev
