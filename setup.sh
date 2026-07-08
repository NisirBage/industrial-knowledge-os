#!/usr/bin/env bash
# One-command bootstrap: creates a backend virtualenv, installs both
# dependency sets, and copies .env templates without overwriting anything
# you've already configured. Safe to re-run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Checking prerequisites"
command -v python3 >/dev/null 2>&1 || { echo "python3 not found. Install Python 3.13+."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "node not found. Install Node.js 20+."; exit 1; }

echo "==> Backend: creating virtualenv (backend/.venv) and installing dependencies"
cd backend
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    created backend/.env (edit it to set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY for real narration)"
else
  echo "    backend/.env already exists, left untouched"
fi
deactivate
cd ..

echo "==> Frontend: installing dependencies"
cd frontend
npm install --silent
if [ ! -f .env ]; then
  cp .env.example .env
  echo "    created frontend/.env"
else
  echo "    frontend/.env already exists, left untouched"
fi
cd ..

echo ""
echo "Setup complete. Next steps:"
echo "  ./run-dev.sh          # start backend (:8000) + frontend (:5173)"
echo "  docker compose up -d  # optional: real Qdrant + Neo4j (falls back automatically without it)"
echo "  cd backend && source .venv/bin/activate && python scripts/seed_demo_data.py   # optional: seed demo data"
