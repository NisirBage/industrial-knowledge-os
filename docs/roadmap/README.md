# Roadmap

Honest state: this is a hackathon-originated, portfolio-grade platform with
production-quality engineering practices, not a production deployment. What
follows is real future work, not vague aspiration.

## Ingestion

- **OCR** for scanned/image-only documents (not implemented — today's PDF
  loader is text-layer only via `pdfplumber`)
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
- Ranking and graph-ML problem types (currently no default algorithm — see
  [docs/architecture/ml-lab.md](../architecture/ml-lab.md#3-model-training-framework))

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
