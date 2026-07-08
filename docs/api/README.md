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

## Error handling

- `422 Unprocessable Entity` — invalid enum value (e.g. unknown `task_type`),
  with the valid options listed in the error detail.
- `400 Bad Request` — a syntactically valid but not-yet-trainable
  `ml_task_type`.
- `404 Not Found` — unknown resource id (e.g. `model_id`, `experiment_id`).
- `500` is never used to mask a real error being hidden by a fallback —
  degraded dependencies (Qdrant/Neo4j/GLiNER/reranker) fail over automatically
  at startup instead; see [docs/architecture/deployment.md](../architecture/deployment.md).
