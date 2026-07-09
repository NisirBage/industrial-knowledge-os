# IKIOS v1.4.0 Release Notes

**Tag:** `v1.4.0` · **Date:** 2026-07-09 · **Codename:** Retrieval & Knowledge Optimization

## What is IKIOS?

IKIOS (Industrial Knowledge Intelligence OS) is an industrial AI platform
built for the ET AI Hackathon 2026 (Problem Statement 8: "AI for Industrial
Knowledge Intelligence: Unified Asset & Operations Brain"). It ingests
industrial documents *and images*, builds a structured knowledge graph
spanning both, learns from that graph via representation learning, reasons
over evidence with explicit contradiction detection, generates justified
recommendations, learns from feedback, trains real scikit-learn and
graph-embedding models, and exposes a full research/benchmarking framework.

## Why this release exists

This is a **pure optimization milestone** — no new end-user functionality.
Every prior milestone's retrieval path (`HybridRetriever`) and planning path
(`TaskGraphExecutor`) stay architecturally exactly what they were; this
release adds optional performance and quality layers behind those same
interfaces. Every new capability — caching, parallel fetch, learning-to-rank,
entity resolution, context compression, explainability — is a constructor
parameter defaulting to `None`/off, so omitting all of them reproduces the
exact pre-1.4.0 behavior. The Planner, Industrial Reasoning Graph,
Recommendation Engine, and Prediction Engine were not touched.

## What's new

- **Multi-Level Caching** (`app/caching/`) — a generic thread-safe
  `LRUTTLCache`, four independent instances (embedding/expansion/reranker/
  result) wired into `HybridRetriever`, with real hit/miss/eviction stats
  exposed at `GET /observability/cache-stats`.
- **Parallel Retrieval Executor** — `HybridRetriever`'s internal graph
  expansion / BM25 / vector search fetch concurrently given an injected
  `Executor`; `TaskGraphExecutor` runs every planner node within a
  dependency round concurrently via the new `TaskGraph.levels()`. Two
  separate thread pools avoid nested-submission deadlock.
- **Learning-to-Rank Retrieval** (`app/retrieval/ranking/`) — a `Ranker`
  Protocol, formula-based RRF ranker, and a real LightGBM LambdaMART ranker
  with an honest RRF fallback when untrained. `POST /ranking/train` builds
  labeled training queries directly from the live Knowledge Graph and
  hot-swaps a trained ranker into production retrieval, registering it into
  the same `SqlModelRegistry` M6 built.
- **Retrieval Explainability** — every `Evidence` item now carries a real,
  computed `explanation` string and `final_rank`; the Chat UI renders it as
  an expandable "Why was this retrieved?" section per citation.
- **Intelligent Entity Resolution** (`app/entity_resolution/`) — combines
  edit distance, ontology similarity, graph-neighborhood similarity,
  embedding similarity, and acronym expansion into an explainable merge
  decision (`canonical_id`/`aliases`/`confidence`/`merge_reason`).
  `GraphMerger` merges duplicate Knowledge Graph nodes with full provenance
  and supports `rollback()`. New `/entity-resolution/*` endpoints, plus a
  Research Dashboard panel to review candidates and merge with one click.
- **Context Compression Engine** (`app/context_compression/`) — exact +
  near-duplicate removal, adjacent timeline merging, real agglomerative
  semantic clustering, three configurable levels, and `CompressionStats`
  reporting a real compression ratio and token reduction. New
  `POST /context-compression/preview`, plus a Research Dashboard panel
  rendering the real summary string live (verified: "Compressed 14 chunks
  into 2 evidence groups (2 clusters, 0 exact + 0 near duplicates removed,
  ~89% token reduction)" against the real ingested demo dataset).
- **4 new research benchmarks** — `ranking_quality`, `context_compression`,
  `evidence_deduplication`, `entity_resolution` — surfaced in a new
  "Optimization Benchmarks" section of the Research Dashboard.
- **New architecture doc**:
  [docs/architecture/retrieval-optimization.md](docs/architecture/retrieval-optimization.md)
  — every optimization layer with diagrams, real before/after numbers, and
  honest tradeoff analysis.

## Verified live, not just unit-tested

The full backend suite was re-run to completion multiple times (580 → 584 →
651 → 702 → 712 passing) to distinguish a handful of intermittent,
pre-existing environmental test flakes (unrelated vision-benchmark tests
under full-suite load) from real regressions — none were found. A repeated
identical chat query was proven to hit the real `retrieval_result` cache via
a live API call, not just a mocked unit test, and a trained LTR ranker was
proven to actually change chat retrieval's evidence ordering end-to-end via
`GET /ranking/status` + a live `/chat` call. With the demo dataset ingested
into a fresh backend, the Research Dashboard's new panels were exercised
live in the browser too: a duplicate-node merge ("Failure Mode" → "Failure
mode") was clicked through and confirmed removed from a follow-up
`/entity-resolution/duplicates` call, and a context-compression preview
against the real ingested evidence rendered the exact "Compressed 14 chunks
into 2 evidence groups..." explanation string on the page.

## Numbers

- **825 tests pass** (712 backend + 113 frontend, up from 552 backend / 107
  frontend at v1.3.0), zero regressions. Spec required beyond 750 total
  (O1) and beyond 700 (O1.2) — both cleared.
- Real, unedited optimization-benchmark results are published in
  [docs/benchmarks/README.md](docs/benchmarks/README.md#optimization-benchmarks-o1--o12):
  RRF MRR 0.4583 vs. LTR MRR 0.5625, 14→10 evidence compression (0.7143
  ratio, ~28.6% token reduction), deduplication precision 1.0 / recall 0.8,
  entity resolution precision 1.0 / recall 1.0 — including an honest note
  that the ranking-quality win uses a deliberately strong synthetic signal
  and isn't a claim that LTR beats RRF on real, noisier queries.

## Not changed

The Knowledge Graph schema, the 8-stage reasoning orchestrator, the Planner,
the Industrial Reasoning Graph, the Recommendation Engine, the Prediction
Engine, and every existing API route's request/response shape are
untouched. Every new field on `Evidence` (`ltr_score`) and the chat schema
(`final_rank`, `explanation`) is additively defaulted, so old clients
deserializing the response are unaffected.

## Known limitations, stated plainly

- `NeuralRanker` is an architecture-only stub (raises `NotImplementedError`)
  — no ColBERT/SPLADE/RankT5/MonoT5 implementation yet.
- LTR ranking features are limited to what's genuinely computed today
  (`bm25_score`, `vector_score`, `graph_score`, `reranker_score`,
  `fusion_score`, `text_length`) — several spec-named features (ontology
  similarity, OCR confidence, temporal recency) have no real backing data
  yet and are deliberately absent rather than fabricated.
- Timeline compression merges *adjacent* graph-sourced evidence sharing an
  entity id — a structural proxy, since `Evidence` carries no real
  timestamp yet.
- Merge history (`app.state.merge_history`) is process-lifetime only, not a
  persisted audit table.
- `GraphMerger.rollback()` does not remove edges redirected onto the target
  during merge (they may be shared with other merges) — a documented scope
  choice, not a hidden bug.
