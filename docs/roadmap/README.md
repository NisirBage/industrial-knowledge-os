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
