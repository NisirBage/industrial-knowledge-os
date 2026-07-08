#!/usr/bin/env bash
# Production-style run: no auto-reload, multiple backend workers, frontend
# built once and served as static assets. This is "production-lite" for
# demo/portfolio purposes, not a hardened deployment — see
# docs/architecture/deployment.md for what a real production setup adds
# (Postgres, managed Qdrant/Neo4j, secrets management, load balancing).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

cleanup() {
  echo ""
  echo "Stopping..."
  [ -n "${BACKEND_PID:-}" ] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

WORKERS="${WORKERS:-4}"

echo "==> Building frontend"
cd frontend
npm run build
cd ..

echo "==> Starting backend on :8000 ($WORKERS workers, no reload)"
(
  cd backend
  source .venv/bin/activate 2>/dev/null || true
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS"
) &
BACKEND_PID=$!

sleep 1
echo "==> Serving frontend/dist on :4173 (Ctrl+C stops both)"
cd frontend
npx vite preview --port 4173 --host
