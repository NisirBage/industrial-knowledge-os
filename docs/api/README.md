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
`prediction`, `timeline`, plus 7 vision benchmarks added in M7 — see the
[Vision section](#vision-m7) below.

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

## Vision (M7)

Prefix: `/vision`

| Method | Path | Description |
| --- | --- | --- |
| POST | `/vision/images` | Multipart upload — `file` + form fields `source_type`, optional `equipment_hint`. Validates and stores the image, returns its metadata. |
| GET | `/vision/images` | List all ingested vision images |
| GET | `/vision/images/{image_id}` | Fetch one image's metadata |
| GET | `/vision/images/{image_id}/file` | Serves the raw stored image bytes (added in M7.1 for the Vision Workspace's Interactive Viewer) |
| POST | `/vision/images/{image_id}/analyze` | Runs OCR + diagram parsing + equipment detection + gauge reading + defect detection, writes every result into the Vision Knowledge Graph, and returns them for the Vision Workspace to render as overlays |
| POST | `/vision/query` | `{"query": "...", "top_k": 8}` → runs `MultimodalRetriever`, returning merged text + visual `Evidence` |

Valid `source_type` values: `inspection_photo`, `equipment_photo`,
`pid_diagram`, `electrical_schematic`, `engineering_drawing`,
`maintenance_image`, `handwritten_note` (unsupported → `422`). Analyzing an
unknown `image_id` → `404`. `VisionDocumentOut` includes `size_bytes`; every
item in `VisionAnalysisOut`'s five result arrays includes a `graph_node_id`
(added in M7.1) linking it back to the exact node
`VisionKnowledgeGraphBuilder` created for it. See
[docs/architecture/vision.md](../architecture/vision.md).

New auto-runnable benchmark names for `POST /research/benchmarks/{name}/run`
(see the Research section above): `vision_ocr`, `vision_diagram_parsing`,
`vision_equipment_recognition`, `vision_gauge_reading`,
`vision_defect_detection`, `vision_multimodal_retrieval`, `vision_latency`.

## Graph Intelligence (M9)

Prefix: `/graph-intelligence`

| Method | Path | Description |
| --- | --- | --- |
| GET | `/graph-intelligence/summary` | Node/edge counts, distinct node and relation types |
| GET | `/graph-intelligence/graph` | Full node-link export (with community id) for the Graph Intelligence workspace |
| GET | `/graph-intelligence/features/{equipment_name}` | Structural (degree, PageRank, betweenness, community) + domain (maintenance, incidents, defects, prediction confidence) features for one node |
| GET | `/graph-intelligence/embeddings/models` | Available embedding model names: `node2vec`, `deepwalk`, `gcn`, `graphsage`, `gat` |
| POST | `/graph-intelligence/embeddings/train` | `{"model_name": "...", "dimensions": 32}` → trains and caches an embedding, registers it in the shared model registry |
| GET | `/graph-intelligence/similarity/{equipment_name}?top_k=5` | Nearest neighbors by cosine similarity over the currently cached (or latest registered) embedding |
| GET | `/graph-intelligence/explain/{equipment_name}` | Nearest neighbors, important edges, community, supporting evidence, confidence, and a human-readable summary |
| POST | `/graph-intelligence/link-prediction/candidates` | `{"equipment_name": "...", "top_k": 5}` → ranked link-prediction candidates with confidence and explanation |
| POST | `/graph-intelligence/node-classification/train` | `{"task_type": "failure_risk", "equipment_names": [...]}` → trains a graph-embedding model against the rule baseline for that task, registers it, returns feature importance |
| POST | `/graph-intelligence/propagation/simulate` | `{"equipment_name": "...", "propagation_type": "failure", "max_depth": 3}` → simulated risk propagation with per-step risk and path |

Similarity/explain/link-prediction/node-classification all require an
embedding to have been trained first (`POST /embeddings/train`) — otherwise
`400`. Unknown equipment names → `404`. See
[docs/architecture/graph-intelligence.md](../architecture/graph-intelligence.md).

New auto-runnable benchmark names for `POST /research/benchmarks/{name}/run`:
`graph_embedding_node2vec`, `graph_embedding_deepwalk`,
`graph_embedding_gcn`, `graph_embedding_graphsage`, `graph_embedding_gat`,
`graph_link_prediction`, `graph_node_classification`.

## Ranking (O1)

Prefix: `/ranking`

| Method | Path | Description |
| --- | --- | --- |
| POST | `/ranking/train` | Builds labeled `(query, relevant_document_ids)` pairs directly from the live Knowledge Graph, trains a LightGBM LambdaMART ranker, registers it (`task_type="retrieval_ranking"`), and hot-swaps it into the live `HybridRetriever`. Returns feature importance. `400` if fewer than 2 labeled queries produce candidates. |
| GET | `/ranking/status` | Whether a trained ranker is currently active, and its registry metadata if so |

## Entity Resolution (O1.2)

Prefix: `/entity-resolution`

| Method | Path | Description |
| --- | --- | --- |
| GET | `/entity-resolution/duplicates?threshold=0.82` | Candidate duplicate node pairs from the live Knowledge Graph, each with per-signal scores and a merge reason |
| POST | `/entity-resolution/merge` | `{"source_id": "...", "target_id": "..."}` → merges the source node into the target (edges redirected, properties merged, provenance recorded), returns the `MergeRecord` |
| POST | `/entity-resolution/rollback` | `{"merge_id": "..."}` → recreates the source node and its original edges from the recorded `MergeRecord` |

Merge history lives in `app.state.merge_history` (process-lifetime, not a
persisted audit table — see
[docs/architecture/retrieval-optimization.md](../architecture/retrieval-optimization.md#o12-component-12--entity-resolution--canonical-knowledge-graph)).

## Context Compression (O1.2)

Prefix: `/context-compression`

| Method | Path | Description |
| --- | --- | --- |
| POST | `/context-compression/preview` | `{"query": "...", "level": "balanced"}` → retrieves real evidence via the live `HybridRetriever`, then compresses it (dedup → timeline merge → semantic clustering), returning the compressed bundle and `CompressionStats` (ratio, token reduction, duplicate/cluster counts). `level` is one of `light`/`balanced`/`aggressive`. |

This is a preview/benchmarking surface — the live Chat pipeline does not
call it; compression stays optional per the O1.2 architectural constraint.

## Observability additions (O1)

| Method | Path | Description |
| --- | --- | --- |
| GET | `/observability/cache-stats` | Real hit/miss/eviction/hit-ratio counters for each of the four `LRUTTLCache` instances (`retrieval_embedding_cache`, `retrieval_expansion_cache`, `retrieval_reranker_cache`, `retrieval_result_cache`) |

New auto-runnable benchmark names for `POST /research/benchmarks/{name}/run`
(see the Research section above): `ranking_quality`, `context_compression`,
`evidence_deduplication`, `entity_resolution`.

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
