# API Reference

Base URL: `http://localhost:8000/api/v1`. Interactive OpenAPI docs are always
available at `/docs` (Swagger UI) and `/redoc` while the backend is running.

All request/response bodies are validated Pydantic schemas — see
[`backend/app/schemas/`](../../backend/app/schemas) for exact field types.

## Health

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness check |

## Documents (ingestion)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/documents/upload` | Multipart file upload — runs the full ingestion pipeline synchronously (chunk → embed → extract entities/relations → update graph) |
| GET | `/documents` | List ingested documents with chunk/entity counts |

Supported extensions: `.pdf`, `.txt`, `.md` (unsupported → `422`). Max upload
size: `MAX_UPLOAD_SIZE_MB` (default 25MB; oversized → `413`) — see
[docs/getting-started/configuration.md](../getting-started/configuration.md).

## Chat (reasoning)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/chat` | `{"query": "..."}` → runs the 8-stage LangGraph pipeline, returns a cited answer, evidence, contradictions, recommendations, and per-stage latencies |

See [docs/architecture/reasoning.md](../architecture/reasoning.md) for what
happens inside.

## Timeline (M4)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/equipment/{canonical_name}/timeline?view=full\|equipment\|incident\|maintenance` | Equipment event history from the knowledge graph |

## Predictions (M4)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/predictions` | `{"equipment_id": "P-101", "task_type": "failure_risk"}` → runs the currently-registered model (rule-based, or a trained ML model if one has been trained via the ML Lab) for that task type |

Valid `task_type` values: `remaining_useful_life`, `failure_risk`,
`inspection_priority`, `maintenance_priority`, `recommendation_priority`,
`missing_documentation_risk`, `compliance_risk`.

## Feedback (M4)

| Method | Path | Description |
| --- | --- | --- |
| POST | `/feedback` | Record recommendation accepted/rejected/modified, or an entity/relation/recommendation correction |

## Observability (M4)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/observability/metrics` | Aggregate platform metrics: feedback volume, average prediction confidence, average prediction/feature-generation latency |

## Research (M5)

Prefix: `/research`

| Method | Path | Description |
| --- | --- | --- |
| GET | `/research/experiments` | List all recorded experiments (newest first) |
| GET | `/research/experiments/{experiment_id}` | Fetch one experiment |
| POST | `/research/benchmarks/{benchmark_name}/run` | Run a benchmark now; creates an Experiment record |
| GET | `/research/benchmarks/{benchmark_name}/history` | Prior runs of one benchmark |
| POST | `/research/ab-test/entity-extraction` | Regex-only vs. production entity extraction A/B test |
| GET | `/research/calibration` | Confidence calibration buckets (predicted vs. actual accuracy) |

Valid `benchmark_name` values: `entity_extraction`, `relation_extraction`,
`knowledge_graph`, `hybrid_retrieval`, `reasoning`, `recommendation`,
`prediction`, `timeline`.

## ML Lab (M6)

Prefix: `/ml`

| Method | Path | Description |
| --- | --- | --- |
| POST | `/ml/train/{ml_task_type}` | Train a real scikit-learn model on a synthetic dataset; registers it and hot-swaps it into the live Prediction Engine |
| GET | `/ml/models` | List registered models, optional `?task_type=` filter |
| GET | `/ml/models/{model_id}` | Fetch one registered model's metadata |
| GET | `/ml/feature-definitions` | The Feature Store's 16 versioned feature definitions |
| POST | `/ml/drift` | Compute Population Stability Index between two numeric distributions |

Valid `ml_task_type` values with a trainable synthetic generator today:
`failure_prediction`, `maintenance_priority`. See
[docs/architecture/ml-lab.md](../architecture/ml-lab.md#1-training-dataset-builder)
for why the other five `MLTaskType` values return `400` today.

`POST /ml/train/{ml_task_type}` request body (all fields optional):

| Field | Default | Purpose |
| --- | --- | --- |
| `sample_size` | 300 | Synthetic training rows to generate |
| `seed` | 42 | Deterministic dataset + split seed |
| `include_cross_validation` | `false` | Also compute k-fold cross-validated metrics (costs `k` extra model fits) |
| `cv_folds` | 5 | Fold count when `include_cross_validation` is `true` |

The response's `model.metrics` is always a single held-out train/test split;
`cross_validation_metrics`/`cross_validation_fold_count` are `null` unless
`include_cross_validation` was set — the two are never merged into one
number. See [docs/architecture/ml-lab.md](../architecture/ml-lab.md#3-model-training-framework).

## Error handling

Every 4xx response body is `{"detail": "..."}` (FastAPI's default shape),
with the detail message naming valid options wherever the request named an
invalid enum value — never just "invalid input."

| Code | Meaning | Where it's used |
| --- | --- | --- |
| `422 Unprocessable Entity` | Request failed schema validation, OR named an unknown enum value (`task_type`, `kind`, `ml_task_type`), OR an unsupported file extension on upload | Every route with a request body; `/documents/upload` |
| `400 Bad Request` | Syntactically valid request that isn't actionable right now — a not-yet-trainable `ml_task_type`, or a benchmark that needs live data rather than a fresh synthetic dataset | `/ml/train/{ml_task_type}`, `/research/benchmarks/{benchmark_name}/run` |
| `404 Not Found` | Unknown resource id | `/ml/models/{model_id}`, `/research/experiments/{experiment_id}` |
| `405 Method Not Allowed` | Wrong HTTP verb for the path (FastAPI default) | Any route called with an unsupported method |
| `413 Payload Too Large` | Upload exceeds `MAX_UPLOAD_SIZE_MB` (checked while streaming, before the full file is buffered) | `/documents/upload` |
| `500` | Never used to mask a real error behind a fallback — degraded dependencies (Qdrant/Neo4j/GLiNER/reranker) fail over automatically at startup instead; see [docs/architecture/deployment.md](../architecture/deployment.md). A `500` here means an actual unhandled bug — please [report it](../../SUPPORT.md). | N/A by design |

All of the above are covered by automated `TestClient` tests in
`backend/tests/test_api_*.py` — see
[docs/contributing/testing.md](../contributing/testing.md#api-layer-tests-v101).
