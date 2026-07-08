# Changelog

All notable changes to IKIOS are documented here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versioning follows
[Semantic Versioning](https://semver.org/). Entries are grouped by
development milestone rather than individual commits, since each milestone
is the unit of "no regressions, tests pass" verification this project uses.

## [1.3.0] — M9: Industrial Graph Intelligence Platform

Adds graph representation learning underneath the existing deterministic
Knowledge Graph — node embeddings, link prediction, node classification, and
risk propagation — as another reasoning source, not a replacement for
symbolic reasoning. No parallel ML infrastructure: every graph model
registers into and is monitored/explained through the exact same
`SqlModelRegistry`/`Explainer`/`ModelMonitor` M6 built.

### Added
- **`app/graph_intelligence/`** (all new): `GraphDatasetBuilder` (PyG
  homogeneous/heterogeneous, NetworkX, CSV, GraphML, JSON exports,
  deterministic node ordering), `GraphFeatureStore` (structural: degree/
  PageRank/betweenness/community; domain: maintenance/incidents/defects/
  OCR confidence/prediction confidence/recommendation acceptance),
  `Node2VecEmbedding`/`DeepWalkEmbedding`/`GCNEmbedding`/
  `GraphSAGEEmbedding`/`GATEmbedding` behind shared `NodeEmbeddingModel`/
  `EdgeEmbeddingModel`/`GraphEmbeddingModel` Protocols (no `torch_geometric`
  coupling outside this module), `LinkPredictionTrainer`/`LinkPredictor`,
  `GraphNodeClassificationTrainer`/`GraphPredictionModel` (trains against
  the real rule-based model's own output — no fabricated labels),
  `GraphPropagationEngine` (5 propagation types, decayed BFS, merges into
  the M3 Industrial Reasoning Graph), `GraphSimilarityService`,
  `GraphExplanationService`, and the top-level `GraphEmbeddingTrainer`/
  `GraphInferenceEngine` orchestration classes.
- **Graph Intelligence API** (`/api/v1/graph-intelligence/*`): summary,
  full graph export, per-node features, embedding training, similarity
  search, explanation, link-prediction candidates, node-classification
  training (with feature importance), risk propagation simulation.
- **Graph Intelligence Workspace** (`frontend/src/pages/
  GraphIntelligencePage.tsx`): interactive Cytoscape.js graph (zoom, node
  type/name filtering, community coloring, click-to-select), embedding
  training, similarity search with explanations, link prediction, risk
  propagation, node classification training with a live model registry
  table (tagged Graph vs. Classical), and on-demand research benchmarks.
- **7 new research benchmarks**: `graph_embedding_node2vec/deepwalk/gcn/
  graphsage/gat`, `graph_link_prediction` (real train/test split — held-out
  edges physically removed before training), `graph_node_classification`
  (rule baseline vs. graph model, honestly reported either way).
- **ML Laboratory integration**: node-classification training now surfaces
  real `Explainer.feature_importance()` output for the graph model's
  estimator, matching classical models' training response shape.

### Changed
- `GraphRepository`/`InMemoryGraphRepository` gained `all_nodes()`/
  `all_edges()` for full-graph export.
- `LearningRepository` gained `latest_prediction_confidence()`.
- `GraphPredictionModel` exposes a public `trained_model` property instead
  of requiring callers to reach into a private attribute.

### Fixed (discovered during this milestone, not present in a shipped release)
- A Windows-specific `neo4j`-before-`torch` import order reliably broke
  torch's native DLL init — fixed by importing `torch` first in `main.py`
  and `conftest.py`.
- `GraphDatasetBuilder.to_networkx()` crashed when a node's `properties`
  dict itself contained a `confidence` key (common for M7 vision nodes).
- The frontend node-classification handler unconditionally refreshed the
  model registry list even when no model was registered, crashing on an
  undefined response — caught by a new `GraphIntelligencePage` test.

### Testing
- Backend: 552 tests (up from 523). Frontend: 107 tests (up from 66).
  **Total: 659 automated tests** (spec required >600). Zero regressions.

## [1.2.0] — M7.1: Vision Workspace (Frontend)

Completes M7 by building the Vision Workspace frontend — the 10th and final
component of the Multimodal Vision Intelligence Platform milestone. No
backend architecture changes beyond minor, additive API fixes discovered
during integration (per the milestone's explicit scope constraint).

### Added
- **Vision Workspace** (`frontend/src/pages/VisionPage.tsx`): a single-page
  workspace covering upload → analysis → interactive viewer → Knowledge
  Graph → multimodal query → research benchmarks, following the existing
  four-workspace design language exactly (Tailwind card pattern, `loading`/
  `error` state convention, no router/state library).
  - Drag-and-drop + click-to-browse upload (PNG/JPEG/BMP/TIFF), with source
    type and optional equipment hint.
  - Interactive viewer: SVG bounding-box overlays (color-coded by category),
    per-layer visibility toggles, zoom (50%-300%) with scroll-based pan,
    native hover tooltips, click-to-select synchronized across the viewer,
    result lists, and the Knowledge Graph panel.
  - Analysis pipeline: runs OCR/diagram/detection/gauge/defect detection via
    the existing `/vision/images/{id}/analyze` endpoint, shows a completion
    checklist.
  - Knowledge Graph panel: one row per created graph node with its real
    `graph_node_id`, confidence, and type.
  - Multimodal Query panel: asks a question, shows evidence with source
    badges distinguishing Visual/Graph/Text provenance.
  - Research panel: runs any of the 7 vision benchmarks on demand, shows
    metrics and latency.
- **Two minor, additive backend API fixes** discovered during frontend
  integration (explicitly permitted by this milestone's scope):
  - `GET /vision/images/{id}/file` — serves the stored image bytes (nothing
    previously served the image itself, only metadata/analysis about it).
  - `graph_node_id` on every `VisionAnalysisOut` result item, `size_bytes` on
    `VisionDocumentOut` — both needed real data the frontend's Component 6/8
    requirements ("highlight graph node," "show size") depended on.
- **Frontend test infrastructure, built from scratch**: Vitest +
  React Testing Library + `@testing-library/user-event` + jsdom (no frontend
  tests existed before this release). 66 new frontend tests
  (`api.vision.test.ts`, `VisionPage.test.tsx`) covering upload, drag-and-
  drop, analysis, overlay rendering/selection/zoom, the Knowledge Graph
  panel, multimodal query, and research benchmarks.
- 3 new backend regression tests for the two API additions above.
- **510 total tests** (444 backend + 66 frontend), zero regressions.

## [1.1.0] — M7: Multimodal Vision Intelligence Platform

Evolves IKIOS from document-centric to multimodal: images (inspection
photos, equipment photos, P&IDs, electrical schematics, engineering
drawings, maintenance images, handwritten notes) become first-class
documents, reasoned over jointly with text through the same Knowledge Graph
and Evidence Fusion pipeline. No changes to the M3 chat orchestrator.

### Added
- **Vision Ingestion Framework** (`app/vision/ingestion/`): validates
  (Pillow), stores, and tracks uploaded images via a new
  `VisionMetadataRepository` (own database, `VISION_DB_URL`).
- **Industrial OCR** (`app/vision/ocr/`): `EasyOCRProvider` (real, pip-only,
  no system binary) with bounding boxes and confidence per region;
  `UnavailableOCRProvider` fallback if the model can't load.
- **Engineering Diagram Understanding** (`app/vision/diagram/`): classical-CV
  valve/pump/pipe detection (Hough circles, contour approximation, Hough
  lines) — see [docs/architecture/diagram-parser.md](docs/architecture/diagram-parser.md)
  for the honest precision/recall discussion.
- **Equipment Recognition** (`app/vision/detection/`): region-proposal
  baseline (Otsu threshold + contours) behind a pluggable `ObjectDetector`
  Protocol, ready for a future YOLO/GroundingDINO/RT-DETR/Detectron2 swap-in.
- **Gauge & Instrument Reading** (`app/vision/gauge/`): detect → read →
  validate staged pipeline; needle angle → normalized 0.0-1.0 dial-scale
  value (explicitly not a calibrated engineering unit).
- **Defect Detection** (`app/vision/defect/`): real detectors for corrosion
  (HSV color-range) and cracks (edge-elongation); the other 6 documented
  defect types are explicitly architecture-only, not faked.
- **Vision Knowledge Graph**: 8 new `EntityLabel` + 8 new `RelationType`
  values; every OCR/diagram/detection/gauge/defect result becomes a real
  graph node through the existing `GraphRepositoryProtocol`.
- **Multimodal Retrieval** (`app/vision/retrieval/multimodal.py`): merges
  text and visual evidence into one list via the existing `Evidence`
  dataclass (`source="vision:*"`) — zero changes to `EvidenceFusionEngine`,
  `ContradictionDetector`, or `ReasoningGraphBuilder`. Degrades gracefully
  (logs + continues) if text retrieval fails, matching
  `TaskGraphExecutor`'s existing per-node error handling.
- **Vision API** (`app/api/v1/vision.py`): `POST /vision/images`,
  `GET /vision/images`, `GET /vision/images/{id}`,
  `POST /vision/images/{id}/analyze`, `POST /vision/query`.
- **7 new vision benchmarks** integrated with the existing
  `ExperimentManager` (no parallel research framework): OCR quality, diagram
  parsing, equipment recognition, gauge reading, defect detection,
  multimodal retrieval recall, and vision-pipeline latency.
- **`VisionModel`/`VisionFeatureExtractor` abstractions** (`app/vision/abstractions.py`)
  completing the 7-interface model abstraction guide the milestone specified.
- **Vision demo dataset** (`vision_demo_data/`): 4 deterministic synthetic
  images with exact ground truth, honestly documenting a real classical-CV
  limitation (composite-scene pump detection) rather than hiding it.
- 74 new backend tests (367 → 441 passing); coverage held at 96%.

### Fixed
- **`GraphRepository.create_node()` (Neo4j)** silently dropped its
  `properties` argument — discovered while building the Vision Knowledge
  Graph. Fixed, plus a new `get_node_properties()` method on
  `GraphRepositoryProtocol`.

## [1.0.1] — Hardening Sprint

Fixes every finding from an independent Principal AI Architect audit
performed against v1.0.0. No new AI capability; no breaking API changes.

### Fixed
- **Contradiction detection precision**: rewrote `ContradictionDetector` to
  associate tag mentions with the sentence they appear in (not the whole
  evidence chunk) and deduplicate by `(tag, antonym pair)` instead of by
  chunk-pair combination — a single real disagreement repeated across
  duplicate/near-duplicate evidence no longer reports dozens of
  "contradictions." 5 new regression tests.
- **Unsupported upload file types** now return `422` with the supported
  extension list instead of an unhandled `500`.
- **Oversized uploads** now return `413` — `/documents/upload` streams and
  checks size against a configurable `MAX_UPLOAD_SIZE_MB` (default 25MB)
  before buffering the full file.

### Added
- **HTTP/API-layer test suite**: 49 new `TestClient` tests
  (`backend/tests/test_api_*.py`, shared session-scoped fixture in
  `tests/conftest.py`) covering every router's happy and error paths.
  Backend coverage: 75% → 96%.
- **Indirect prompt-injection hardening**: document-derived content reaching
  the LLM narrator (`app/narration/narrator.py`) and recommendation
  justification refinement (`app/recommendation/engine.py`) is now wrapped
  in `<retrieved_evidence>`/`<justifications>` delimiters, with both system
  prompts explicitly instructing the model never to follow instructions
  embedded in that content. 4 new regression tests verifying the delimiter
  framing and system-prompt language.
- **Cross-validation metrics**: `SklearnTrainer.cross_validate()` (k-fold,
  stratified for classification), exposed as an opt-in
  `include_cross_validation`/`cv_folds` field on `POST /ml/train/{task}`,
  returned as a separate `cross_validation_metrics` field — never merged
  with the held-out split's `model.metrics`. 7 new tests.
- **Concurrent startup**: the five independent dependency checks (embedder,
  Qdrant, Neo4j, GLiNER, reranker) in `lifespan()` now run concurrently via
  `asyncio.to_thread` + `asyncio.gather` instead of sequentially. Measured
  improvement: ~20s vs. ~25-31s in this environment (modest, not the
  larger theoretical best case — see
  [docs/architecture/deployment.md#startup-performance](docs/architecture/deployment.md#startup-performance)).

### Changed
- `docs/api/README.md`'s error-handling section now documents every 4xx/5xx
  code with its trigger condition, including the new `413`.
- `docs/benchmarks/README.md` leads with an explicit methodology/ground-truth
  section (single-relevant-document-per-query, single-chunk synthetic
  ingestion, environment-dependent results) instead of trailing notes only.
- `SECURITY.md` documents the v1.0.1 hardening and its honest limits (a
  mitigation, not a guarantee, against prompt injection).
- Expanded `backend/.env.example` with `MAX_UPLOAD_SIZE_MB`.

### Testing
367 tests pass (up from 302), zero regressions.

## [1.0.0] — RC1: Enterprise Productization & Open Source Release

No new AI capability — this release turns the platform into a polished,
documented, deployable open-source project.

### Added
- Full documentation portal (`docs/`): getting-started, 11 architecture docs
  with Mermaid diagrams, API reference, contributing guide, FAQ, roadmap
- Deployment toolkit: `setup.ps1`/`setup.sh`, `run-dev.ps1`/`run-dev.sh`,
  `run-prod.ps1`/`run-prod.sh`, expanded `.env.example` covering every
  configuration variable
- Curated demo dataset (`demo_data/`): 7 documents spanning every subsystem,
  hand-labeled `ground_truth.json`, seeding script
  (`backend/scripts/seed_demo_data.py`)
- Real benchmark reports (`docs/benchmarks/`) generated by
  `backend/scripts/generate_benchmark_reports.py`: entity/relation extraction,
  knowledge graph, hybrid retrieval, reasoning, recommendation, prediction,
  timeline, plus dedicated comparisons (regex vs GLiNER, vector vs hybrid vs
  graph-aware retrieval, rule-based vs ML prediction)
- GitHub Actions CI (lint, tests with coverage, frontend build)
- Repository quality files: `LICENSE` (MIT), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, issue/PR templates
- `RELEASE_NOTES.md` and `docs/RELEASE_REPORT.md` (full M0-M6 technical history)

### Changed
- Rewrote `README.md` as a comprehensive project overview
- Expanded `backend/.env.example` with previously-undocumented settings
  (`QDRANT_LOCAL_PATH`, `LEARNING_DB_URL`, `EXPERIMENT_DB_URL`,
  `ML_REGISTRY_DB_URL`, `ML_MODELS_DIR`) — no default values changed

### Fixed
- Minor lint cleanup (unused imports, import ordering) surfaced by adding
  `ruff` — no behavioral changes

## [0.6.0] — M6: Industrial Machine Learning Laboratory

### Added
- Training Dataset Builder, Feature Store (16 versioned features), Model
  Training Framework (scikit-learn, pluggable algorithm factories), Model
  Registry (SQLite + joblib), Inference Engine (`MLPredictionModel`,
  runtime-swappable into the M4 Prediction Engine), Explainability Layer
  (feature importance, permutation importance, calibration curves, decision
  paths), Model Monitoring (PSI-based drift detection)
- ML Laboratory frontend workspace
- 74 new tests (301 total)

## [0.5.0] — M5: AI Research, Benchmarking & Experimentation Framework

### Added
- Experiment Manager with git-commit auto-capture, 8 Benchmark Runners,
  Synthetic Dataset Generator with noise injection, A/B Evaluation Framework
  (paired t-test), Report Exporter (JSON/CSV/Markdown/HTML), Offline
  Evaluation Harness (memory/CPU profiling via stdlib)
- Research Dashboard frontend workspace
- 77 new tests (227 total)

## [0.4.0] — M4: Industrial Learning & Prediction Engine

### Added
- Timeline Engine, rule-based Feature Engineering, Prediction Engine
  (pluggable `PredictionModel` protocol), Feedback Engine, Evaluation
  Framework, Learning Repository
- 100+ new tests (150 total)

## [0.3.0] — M3: Industrial Cognitive Engine

### Added
- Task Graph Planner + Executor, Evidence Fusion Engine, Evidence Graph,
  Industrial Reasoning Graph, Contradiction Detector, Recommendation Engine,
  LLM Narrator
- 8-stage LangGraph orchestrator replacing the earlier simple pipeline
- 50+ new tests

## [0.2.5] — M2.5: Industrial Ontology

### Added
- 24-label entity ontology with explicit is-a hierarchy, 15 relation types
- Entity normalizer reconciling GLiNER and regex extraction into one
  canonical graph node per real-world entity

## [0.2.0] — M2: Knowledge Graph & Intelligent Entity Extraction

### Added
- GLiNER zero-shot NER + regex extraction, rule-based relation extraction,
  Knowledge Graph Builder, graph-aware hybrid retrieval (BM25 + vector +
  graph expansion + cross-encoder reranking)

## [0.1.0] — M1: Document Intelligence + M0: Infrastructure

### Added
- PDF/TXT/MD ingestion, chunking, embeddings (Qdrant), FastAPI backend,
  React frontend, Docker Compose (Qdrant + Neo4j), graceful fallback for
  every optional dependency
