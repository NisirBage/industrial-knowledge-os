# Installation

## Prerequisites

| Tool | Version | Required? |
| --- | --- | --- |
| Python | 3.13+ | Yes |
| Node.js | 20+ | Yes |
| Docker | any recent | No — see [fallback modes](#no-docker-thats-fine) |

## Option A — scripted (recommended, under 5 minutes)

```bash
git clone https://github.com/NisirBage/industrial-knowledge-os.git
cd industrial-knowledge-os

# macOS / Linux
./setup.sh

# Windows
.\setup.ps1
```

This installs backend + frontend dependencies, copies `.env.example` → `.env` in
both `backend/` and `frontend/` (only if `.env` doesn't already exist — your
existing config is never overwritten), and prints next steps.

Then start everything:

```bash
# macOS / Linux
./run-dev.sh

# Windows
.\run-dev.ps1
```

This starts the backend (`:8000`) and frontend (`:5173`) together. Open
http://localhost:5173.

## Option B — manual

### 1. Infrastructure (optional)

```bash
docker compose up -d
```

Starts Qdrant (`:6333`) and Neo4j (`:7474` browser / `:7687` bolt).

#### No Docker? That's fine.

The backend detects unreachable services at startup and falls back automatically:

- **Qdrant unreachable** → embedded local-file vector store (`./.qdrant_local`)
- **Neo4j unreachable** → in-memory knowledge graph (not persisted across restarts)
- **GLiNER model unavailable** → regex-only entity extraction
- **Cross-encoder reranker unavailable** → passthrough (fusion rank order preserved)

Every fallback is logged as a `WARNING` at startup — nothing fails silently. Chat,
retrieval, and reasoning all still work fully in fallback mode; only knowledge
graph persistence and neural entity extraction quality are reduced. See
[docs/architecture/deployment.md](../architecture/deployment.md) for the full
fallback matrix.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env      # Windows: copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Runs with `LLM_PROVIDER=mock` by default — no API key needed, deterministic
offline-safe responses. Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in
`.env` for real generated narration.

Verify: `curl http://localhost:8000/api/v1/health`

### 3. Frontend

```bash
cd frontend
npm install
cp .env.example .env      # Windows: copy .env.example .env
npm run dev
```

Open http://localhost:5173.

## Running the tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q
```

301 tests should pass. See [docs/getting-started/quickstart.md](quickstart.md) to
try the app itself, or [docs/contributing/testing.md](../contributing/testing.md)
for the test suite's structure.

## Seeding demo data

```bash
cd backend
python scripts/seed_demo_data.py
```

Ingests the curated [demo dataset](../../demo_data/README.md) so Chat, Timeline,
Predictions, and the Knowledge Graph all have real content to explore immediately
— see [Quick Start](quickstart.md).
