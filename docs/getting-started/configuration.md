# Configuration

All settings live in `backend/.env` (copy from `backend/.env.example`) and are
loaded via Pydantic Settings in
[`app/core/config.py`](../../backend/app/core/config.py). Every field has a
working default — nothing below is required to run IKIOS in mock/fallback mode.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `mock` | `mock` (offline, deterministic, no key) or `anthropic` |
| `ANTHROPIC_API_KEY` | *(empty)* | Required only if `LLM_PROVIDER=anthropic` |
| `LLM_MODEL` | `claude-sonnet-5` | Model id passed to the Anthropic client |
| `QDRANT_URL` | `http://localhost:6333` | Vector store endpoint |
| `QDRANT_COLLECTION` | `ikios_chunks` | Qdrant collection name |
| `QDRANT_LOCAL_PATH` | `./.qdrant_local` | Embedded-mode fallback path when `QDRANT_URL` is unreachable |
| `NEO4J_URI` | `bolt://localhost:7687` | Knowledge graph endpoint |
| `NEO4J_USER` / `NEO4J_PASSWORD` | `neo4j` / `ikios_dev_password` | Neo4j credentials |
| `METADATA_DB_URL` | `sqlite+aiosqlite:///./ikios_metadata.db` | Document/chunk metadata store |
| `LEARNING_DB_URL` | `sqlite+aiosqlite:///./ikios_learning.db` | Predictions, feedback, model performance (M4) |
| `EXPERIMENT_DB_URL` | `sqlite+aiosqlite:///./ikios_experiments.db` | Experiment tracking (M5) |
| `ML_REGISTRY_DB_URL` | `sqlite+aiosqlite:///./ikios_ml_registry.db` | Trained model registry (M6) |
| `ML_MODELS_DIR` | `./.ml_models` | Serialized (`joblib`) model files |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model; falls back to a hashing embedder if unavailable |
| `GLINER_MODEL` | `urchade/gliner_mediumv2.1` | Zero-shot NER model; falls back to regex-only extraction if unavailable |
| `GLINER_THRESHOLD` | `0.4` | Minimum GLiNER confidence to accept an entity span |
| `ENABLE_LLM_ENTITY_VALIDATION` | `false` | Optional LLM-based entity validation pass |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker; falls back to passthrough if unavailable |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated list of allowed frontend origins |

Frontend: `frontend/.env` (copy from `frontend/.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VITE_API_BASE_URL` | `http://localhost:8000/api/v1` | Backend API base URL |

## Fallback behavior

Every external dependency is optional at runtime — see
[docs/architecture/deployment.md](../architecture/deployment.md#fallback-matrix)
for the full table of what degrades and how. In short: nothing crashes if
Docker isn't running or a model can't download; the platform logs a warning and
serves a working, if less capable, experience.
