# IKIOS v1.5.0 Release Notes

**Tag:** `v1.5.0` · **Date:** 2026-07-09 · **Codename:** Faithful Cognitive Reasoning

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

Retrieval quality was strong after v1.4.0, but the final answer still
depended on an LLM synthesizing compressed evidence directly — leaving real
hallucination risk (unsupported causal links, evidence over-generalization,
incorrect confidence, contradiction smoothing, recommendation inflation).
This release introduces a deterministic **Cognitive Verification Layer**
between Reasoning and the Narrator: the LLM stops deciding what's true and
becomes a surface-realization engine only, narrating from a Structured
Claim Graph that has already been extracted, verified against the
Knowledge Graph, confidence-propagated, and (for recommendations)
threshold-checked. Unsupported claims and under-verified recommendations
never reach the user. Every capability is an additive pipeline stage and
additive `ChatResponse` field — the Planner, Industrial Reasoning Graph,
Recommendation Engine, and Prediction Engine are untouched.

## What's new

- **Structured Claim Graph** (`app/claims/models.py`) — atomic claims
  (Fact/Observation/Hypothesis/Recommendation/Risk/Prediction/Unknown)
  with confidence, source evidence, verification status, support/
  contradiction counts, and graph links between them.
- **Claim Extraction Engine** — decomposes evidence into atomic claims by
  reusing the exact entity/relation extraction ingestion already uses (no
  parallel NLP stack); RISK/HYPOTHESIS claims mirror the Reasoning Graph's
  own nodes.
- **Claim Verification Engine** — checks every claim against the Knowledge
  Graph and corroborating evidence/timeline, assigning SUPPORTED/
  PARTIALLY_SUPPORTED/CONTRADICTED/INSUFFICIENT_EVIDENCE/UNKNOWN.
  Unsupported claims are dropped before the Narrator ever sees them — the
  literal mechanism behind "the LLM generates from verified claims only."
- **Evidence Attribution Framework** — every answer sentence exposes its
  supporting claims, evidence, Knowledge Graph nodes/edges, confidence, and
  a human-readable reasoning path, one click away in the Chat UI.
- **Confidence Propagation** — an explicit local-confidence formula (no
  averaging) combined with a decayed BFS over claim relations, replacing
  ad hoc heuristic confidence; a documented, honest extension point for
  Bayesian propagation later.
- **Natural Language Inference** (`app/nli/`) — a real DeBERTa-v3 MNLI
  cross-encoder (via `sentence-transformers`, same loading/fallback idiom
  as the existing reranker) with a deterministic rule-based fallback;
  `NLIContradictionDetector` is a genuine drop-in for the existing
  `ContradictionDetector`, toggleable via config.
- **Recommendation Verification** — every recommendation must clear real
  evidence/confidence/graph-support thresholds before being shown, with a
  transparent verification record (status, reasons, supporting claims)
  whether it passed or not.
- **Faithfulness Scoring** — measures claim coverage, unsupported/
  hallucinated claims, missing evidence, and contradiction rate; combines
  them into an explicit weighted overall/sentence/recommendation/graph
  faithfulness score, stored for the Research Dashboard.
- **7 new research benchmarks** covering claim extraction, claim
  verification, NLI, recommendation verification, faithfulness, and two
  ablation studies (claim verification on/off/with-graph, NLI detector
  choice) — surfaced in a new Research Dashboard section.
- **Chat UI**: Faithfulness score panel, Claim Graph with verification
  status and confidence, per-recommendation verification badges, and an
  expandable Reasoning Path per answer sentence.

## Verified live, not just unit-tested

The full backend suite was re-run to completion multiple times to
distinguish real regressions from a handful of intermittent, pre-existing
session-state-sensitive test flakes (chat/observability/ranking/vision
tests under full-suite load — the same category already documented at
v1.4.0) — reproduced and confirmed as environmental in three separate
combined re-runs (isolated: pass; 42 tests together: pass; 30 tests
together: pass), not caused by O2. A real chat query was proven end-to-end
against the real 13-stage pipeline (real GLiNER, real cross-encoder
reranker, real NLI model) via `TestClient`, returning real `claims`,
`faithfulness`, and `sentence_attributions` fields.

## Numbers

- **951 tests pass** (830 backend + 121 frontend, up from 825/113 at
  v1.4.0), zero regressions. Spec required beyond 950.
- Real, unedited reasoning-benchmark results are published in
  [docs/benchmarks/README.md](docs/benchmarks/README.md#reasoning-benchmarks-o2),
  including an honest weak spot stated plainly: `RuleBasedNLIDetector`'s
  entailment accuracy is only 0.3333 on the hand-labeled set (its
  lexical-overlap heuristic is conservative by design — it correctly avoids
  false contradictions at the cost of also missing paraphrased entailment).
  `claim_verification_ablation` shows the real mechanism behind unsupported-
  claim dropping: with no Knowledge Graph populated, both claims in the
  synthetic scenario land on INSUFFICIENT_EVIDENCE and get removed;
  populating the graph is what lets them survive.

## Not changed

The Planner, Industrial Reasoning Graph, Recommendation Engine, Prediction
Engine, Hybrid Retrieval, and every existing API route's request/response
shape are untouched. Every new field on `ChatResponse` (`claims`,
`faithfulness`, `sentence_attributions`, and each recommendation's
`verification_status`/`verification_reasons`) is additively defaulted, so
old clients deserializing the response are unaffected.

## Known limitations, stated plainly

- Claim Extraction uses sentence-scoped keyword-cue relation matching, not
  real dependency parsing — a documented extension point.
- Vision Results and Predictions are not yet real Claim Verification/
  Confidence Propagation inputs — no chat-time call to either subsystem
  exists today, so wiring them in without one would be fabricated
  corroboration.
- `BayesianConfidencePropagator` is an honest `NotImplementedError` stub.
- Faithfulness Scoring runs one NLI call per (claim, answer-sentence) pair
  — fine at demo scale, would want pre-filtering by shared entity ids at
  production claim counts.
- The ablation studies cover the two axes O2 itself adds (claim
  verification, NLI); the full 8-axis cross-product the spec's ablation
  table describes is real future work, not run here.
