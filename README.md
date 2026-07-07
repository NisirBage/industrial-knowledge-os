# Industrial Knowledge Intelligence OS (IKIOS)

AI-powered Industrial Knowledge Intelligence platform — ET AI Hackathon 2026,
Problem Statement 8. See [CLAUDE.md](CLAUDE.md) for full architecture, vision,
and milestone plan.

## MVP scope

Ingestion (PDF/TXT/MD) → chunking → embeddings (Qdrant) + entity extraction
(Neo4j) → hybrid retrieval (BM25 + vector + graph boost) → LangGraph agent
orchestration (Planner → Retrieval → Maintenance → Reasoning) → cited, evidence-backed
chat answers. Frontend: upload + chat UI.

## Run it

### 1. Start infra

```
docker compose up -d
```

Starts Qdrant (`:6333`) and Neo4j (`:7474` browser / `:7687` bolt).

No Docker available? The backend falls back automatically: Qdrant runs in
embedded local-file mode, and the knowledge graph runs in-memory (not
persisted across restarts). Semantic search and chat still work fully; only
graph persistence and cross-session entity boosting are reduced.

### 2. Backend

```
cd backend
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Runs with `LLM_PROVIDER=mock` by default (no API key needed) — set
`LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `.env` for real generated
answers.

Tests: `pytest` (from `backend/`).

### 3. Frontend

```
cd frontend
npm install
copy .env.example .env
npm run dev
```

Open http://localhost:5173 — upload a document, then ask a question in Chat.

## API

- `POST /api/v1/documents/upload` — multipart file upload, runs full ingestion pipeline
- `GET /api/v1/documents` — list ingested documents
- `POST /api/v1/chat` — `{"query": "..."}` → cited, evidence-backed answer
- `GET /api/v1/health`
