# Roadmap

Honest state: this is a hackathon-originated, portfolio-grade platform with
production-quality engineering practices, not a production deployment. What
follows is real future work, not vague aspiration.

## Ingestion

- **OCR for scanned/image-only PDFs** (not implemented — today's PDF loader
  is text-layer only via `pdfplumber`; M7 added real OCR, but only for
  standalone image uploads via `/vision/images`, not scanned PDF pages —
  see [docs/architecture/vision.md](../architecture/vision.md))
- DOCX, XLSX, email, P&ID drawing loaders (the loader registry in
  `app/documents/loaders.py` is designed for this — one new function per
  format)
- Structure-preserving extraction (tables, figures, headings as distinct
  elements, not flattened to paragraph text)

## Storage

- Postgres in place of SQLite for `MetadataRepository`, `LearningRepository`,
  `SqlExperimentTracker`, `SqlModelRegistry` — every one is already behind a
  Protocol, so this is a new implementation + config change
- Redis for caching / session state
- MinIO (or S3) for original file storage (currently local disk)
- Managed Qdrant/Neo4j clusters instead of local/embedded modes

## Knowledge & Reasoning

- The original vision sketched separate Maintenance/Compliance/Incident
  specialist agents; what was actually built is a more general fixed
  8-stage pipeline (Planner → Retrieval → Knowledge Graph → Evidence Fusion →
  Reasoning → Recommendation → Verifier → Report) that reasons over any
  domain the ontology covers, rather than routing to domain-specific agents.
  Revisiting whether specialized agents add value on top of this is open.
- Richer per-chunk metadata (document type, department, date) to activate the
  currently-passthrough `_metadata_filter()` in hybrid retrieval

## ML Laboratory

- Synthetic label generators for the remaining 5 `MLTaskType` values
  (inspection priority, recommendation acceptance, equipment risk, compliance
  risk, root cause classification) once real historical labels exist to
  validate against
- Real SHAP integration (documented extension point on `Explainer`)
- XGBoost/LightGBM/PyTorch trainers implementing the existing `Trainer`
  protocol
- ONNX/TorchScript serialization, MLflow-backed registry (both are additive —
  `serialization_format` is already a plain string field)
- Ranking problem types (graph-ML now has a default algorithm as of M9 — see
  [docs/architecture/graph-intelligence.md](../architecture/graph-intelligence.md))

## Vision

- **Deep learning detectors** behind the existing `ObjectDetector`/
  `DiagramParser`/`DefectDetector` Protocols — YOLO/GroundingDINO/RT-DETR/
  Detectron2 for equipment recognition, a trained P&ID symbol classifier for
  diagram understanding, a real segmentation model for the 6 currently
  architecture-only defect types (oil leakage, damaged insulation, loose
  components, missing bolts, surface wear) plus higher-recall corrosion/crack
  detection than the current HSV/edge heuristics
- **`VisionFeatureExtractor`** (Protocol exists, no implementation) — an
  embedding model (CLIP or a fine-tuned ResNet) for image-similarity search
  or as input to a classifier trained on real labeled inspection photos
- **ML-based gauge reading** replacing only `read_gauge()` (the swap-in seam
  already exists — see [docs/architecture/vision.md](../architecture/vision.md#5-gauge--instrument-reading)),
  plus digital display reading (`GaugeType.DIGITAL_DISPLAY` exists, unimplemented)
- **Additional `VisionSourceType` values**: thermal imaging, drone imagery,
  CCTV snapshots — the enum and ingestion pipeline are already designed for
  this (one new value + a source-specific analysis hint, not a pipeline rewrite)
- **Relevance-ranked multimodal retrieval** — text and visual evidence are
  currently concatenated, not merged by a unified relevance score (see
  [docs/architecture/multimodal-retrieval.md](../architecture/multimodal-retrieval.md#limitations))
- **Real accuracy benchmarks against labeled real-world photos** — today's
  vision benchmarks use small deterministic synthetic images (exact ground
  truth, but not real-world lighting/angle/wear variation)

## Graph Intelligence (M9)

- **Heterogeneous, relation-type-aware GNN layers** — `GraphDatasetBuilder.
  to_pyg_heterogeneous()` already produces the `HeteroData` shape a relation-
  aware convolution (e.g. `HeteroConv`/`RGCN`) would consume; no such layer
  is implemented yet, only homogeneous GCN/GraphSAGE/GAT
- **DGL export** — the spec calls for it; `GraphDataset`'s node/edge shape
  maps directly, but no DGL-specific conversion code exists yet (PyG,
  NetworkX, CSV, GraphML, and JSON are implemented)
- **Real temporal graph learning** — `GraphNode`/`GraphEdge` carry an
  optional `timestamp` field today (architecture only, per the spec's
  explicit scope), with no temporal-GNN consumer
- **SHAP-based graph model explanations** — same documented extension point
  M6's `Explainer` already has for classical models (would return the same
  `list[FeatureImportanceResult]` shape)
- **Real ground-truth node classification labels** — today's graph node
  classifiers train against the existing rule-based model's own output
  (an honest, real comparison target given no ground-truth risk/priority
  labels exist yet); once real historical outcomes exist, the same trainer
  could target them directly
- **Cross-workspace node selection sync** — clicking a node in the Graph
  Intelligence workspace does not yet jump to that equipment in the Vision,
  Documents, or Chat workspaces (each workspace is independently navigable
  today, not yet linked by shared selection state)

## Retrieval & Knowledge Optimization (O1 / O1.2)

- **`NeuralRanker` real implementation** — ColBERT/SPLADE/RankT5/MonoT5
  behind the existing `Ranker` Protocol (currently a documented stub that
  raises `NotImplementedError`)
- **Real graph-embedding similarity as an LTR feature** — needs a
  chunk-to-graph-node resolver; M9's embeddings are keyed by graph node id,
  `Evidence` by chunk id, and no mapping between the two exists yet
- **Per-chunk timestamps** — would enable real (not adjacency-proxy)
  timeline compression and a genuine "temporal recency" ranking feature;
  `app/documents/chunking.py` would need a date field
- **Persisted merge-history audit table** — `app.state.merge_history` is
  process-lifetime only today, not durable across restarts
- **Real production query logs** for a genuine RRF-vs-LTR ranking-quality
  comparison — the current `ranking_quality` benchmark uses a deliberately
  strong synthetic BM25 signal (see
  [docs/benchmarks/README.md](../benchmarks/README.md#optimization-benchmarks-o1--o12))
- **Real tokenizer for compression stats** — `_estimate_tokens()` is a
  word-count proxy today, consistent enough for before/after ratios but not
  exact subword counts

## Faithful Cognitive Reasoning (O2)

- **Real dependency parsing for Claim Extraction** (spaCy-based), replacing
  today's sentence-scoped keyword-cue rule extraction
- **A future LLM-assisted claim extractor** — a documented extension point,
  not implemented, same honest-stub discipline as `NeuralRanker`
- **Vision Results / Predictions as genuine Claim Verification / Confidence
  Propagation inputs** — no chat-time call to the Vision Knowledge Graph or
  Prediction Engine exists today, so wiring either in without a real call
  would be fabricated corroboration
- **`BayesianConfidencePropagator` real implementation** — currently a
  documented `NotImplementedError` stub
- **A ModernBERT NLI model** once one is published with a compatible
  `sentence-transformers` CrossEncoder checkpoint (`TransformerNLIDetector`
  already accepts any such model via `NLI_MODEL`)
- **`RECOMMENDATION`/`PREDICTION` claim materialization** in the Claim
  Graph itself — today Recommendation Verification reads `Recommendation`
  objects directly rather than a parallel claim representation
- **Pre-filtering candidate claims by shared entity ids** before
  Faithfulness Scoring's NLI pass, needed once claim/answer counts grow
  beyond demo scale (today: one NLI call per claim × sentence pair)
- **The full 8-axis ablation cross-product** (Retrieval × Ranking ×
  Compression × Claim Verification × NLI × Recommendations × Confidence ×
  Faithfulness) — today's `claim_verification_ablation`/`nli_ablation`
  benchmarks cover the two axes O2 itself adds, not the full grid

## Platform

- Authentication/authorization (none exists today — single-tenant, trusted
  environment assumption)
- Background job queue for ingestion (currently synchronous per-upload;
  fine at demo scale, a bottleneck at production document volume)
- Multi-user session isolation

## Research & Benchmarking

- Expand the synthetic dataset generator's domain coverage beyond the current
  document types
- Track benchmark results over time in a persistent dashboard beyond the
  Research tab's live-run view

None of the above require re-architecting anything already built — every
extension point above is a documented seam (a `Protocol`, an empty
`_metadata_filter()`, a `serialization_format` string field) precisely so that
adding it later doesn't mean rewriting what's here.
