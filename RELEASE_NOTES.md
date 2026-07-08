# IKIOS v1.0.1 Release Notes

**Tag:** `v1.0.1` · **Date:** 2026-07-08 · **Codename:** Hardening Sprint

## What is IKIOS?

IKIOS (Industrial Knowledge Intelligence OS) is an industrial AI platform
built for the ET AI Hackathon 2026 (Problem Statement 8: "AI for Industrial
Knowledge Intelligence: Unified Asset & Operations Brain"). It ingests
industrial documents, builds a structured knowledge graph, reasons over
evidence with explicit contradiction detection, generates justified
recommendations, learns from feedback, trains real scikit-learn models, and
exposes a full research/benchmarking framework — seven layers, not a single
chatbot wrapper.

## Why this release exists

Following v1.0.0's tag, an independent Principal AI Architect audit was
performed against the released codebase — an adversarial due-diligence
review, not a feature review. It found no Critical issues, but three High
and five Medium findings worth fixing before the project sits in front of an
enterprise pilot, a hiring panel, or the public internet. This release fixes
all eight, adds regression tests for every one of them, and changes no
existing API's behavior. **No new AI capability was added in this release —
correctness, reliability, and trustworthiness only.**

## What changed

- **Contradiction detection precision (was High-severity).** The detector
  previously flagged one contradiction per *pair of evidence chunks* sharing
  a tag with any antonym term anywhere in the chunk — a single real
  disagreement, repeated across a handful of duplicate/near-duplicate
  documents, could report dozens of "contradictions." It's now scoped to the
  sentence a tag is mentioned in and deduplicated by `(tag, antonym pair)`,
  so one real disagreement is reported once, with every supporting chunk
  aggregated into it. 5 new regression tests, including a reproduction of
  the exact demo-dataset scenario that originally surfaced the bug.
- **HTTP/API test coverage (was High-severity).** All 302 prior tests
  validated business logic directly; none exercised the FastAPI route layer.
  49 new `TestClient` tests now cover every router's happy path and error
  paths (validation, unknown enum values, 404s, 413s). Backend coverage went
  from 75% to 96% as a direct result.
- **Indirect prompt-injection hardening (was High-severity).** Document-
  derived text reaching the LLM narrator or recommendation-refinement calls
  is now wrapped in explicit `<retrieved_evidence>`/`<justifications>`
  delimiters, with both system prompts instructing the model to treat that
  content strictly as data, never as instructions. This is a mitigation, not
  a guarantee — see [SECURITY.md](SECURITY.md#hardening-applied-in-v101) for
  the honest scope of what this does and doesn't protect against.
- **Upload robustness (Medium).** `/documents/upload` now validates the file
  extension before processing (`422` for unsupported types, not a raw `500`)
  and enforces a configurable size limit (`MAX_UPLOAD_SIZE_MB`, default
  25MB, checked while streaming — `413` for oversized files, never fully
  buffered in memory first).
- **Cross-validation metrics (Medium).** `TrainingDatasetBuilder`'s CV fold
  utility existed but was never exercised by the live training path — every
  reported metric was a single train/test split with no variance estimate.
  `POST /ml/train/{task}` now accepts an opt-in `include_cross_validation`
  flag returning k-fold-averaged metrics in a clearly separate
  `cross_validation_metrics` field, never merged with the held-out estimate.
- **Startup performance (Medium).** The five independent dependency checks
  (embedder, Qdrant, Neo4j, GLiNER, reranker) now run concurrently instead of
  sequentially. Measured improvement was real but modest (~20s vs. ~25-31s
  in this environment, with one anomalous 38s run) — reported honestly rather
  than as the larger theoretical win a naive analysis would predict; see
  [docs/architecture/deployment.md#startup-performance](docs/architecture/deployment.md#startup-performance).
- **Benchmark methodology (Medium).** [docs/benchmarks/README.md](docs/benchmarks/README.md)
  now leads with an explicit methodology/ground-truth-assumptions section
  (single-relevant-document-per-query, single-chunk synthetic ingestion,
  environment-dependent results) instead of burying it in trailing notes.
- **API error handling standardized.** Every 4xx code now has a documented
  meaning and trigger condition — see
  [docs/api/README.md#error-handling](docs/api/README.md#error-handling).

## Numbers

- **367 tests pass** (302 → 367: 5 contradiction-detector regression tests,
  49 API-layer tests, 6 ML cross-validation tests, plus injection-hardening
  and audit-fix regression tests), zero regressions.
- **96% backend coverage** (up from 75%), almost entirely from the new
  API-layer tests closing the route-handler gap.

## Not changed

No existing API request/response shape changed in a breaking way — new
fields (`cross_validation_metrics`, etc.) are additive and default to
backward-compatible values (`null`/`false`). No new AI capability, no
architecture redesign — this was a hardening sprint by design.

## Full history

See [CHANGELOG.md](CHANGELOG.md) for the structured per-version changelog
and [docs/RELEASE_REPORT.md](docs/RELEASE_REPORT.md) for the full technical
narrative across all milestones (M0 through v1.0.1).
