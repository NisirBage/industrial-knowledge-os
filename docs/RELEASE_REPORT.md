# IKIOS Release Report: The Evolution of an Industrial AI Platform

This is the technical narrative behind IKIOS v1.0.0 — not a changelog (see
[CHANGELOG.md](../CHANGELOG.md) for that), but the story of *why* each
milestone was built the way it was, what was learned, and what tradeoffs were
made deliberately rather than by accident.

---

## M0 — Infrastructure

**Goal:** a foundation that could grow without rewrites.

Every storage dependency was designed behind a `Protocol` from day one —
Qdrant, Neo4j, and Postgres were all "target" infrastructure, but the MVP
needed to run without any of them installed. The decision made here that
shaped everything after: **build the fallback path first, the real backend
second.** `build_vector_repository()` and `build_graph_repository()` try the
real service and fall back to an embedded/in-memory implementation
automatically. This one decision is why every later milestone could be
demoed on a laptop with zero external services running.

## M1 — Document Intelligence

**Goal:** working retrieval, no graph yet.

PDF/TXT/MD ingestion, paragraph-aware chunking, embeddings into Qdrant (or its
embedded fallback). The honest scope decision here: OCR, DOCX, and structure-
preserving extraction were explicitly deferred, not implied as done — a
pattern that held for the rest of the project (see
[docs/architecture/ingestion.md](architecture/ingestion.md)).

## M2 — Knowledge Graph & Intelligent Entity Extraction

**Goal:** move beyond "search over chunks" to structured knowledge.

GLiNER (zero-shot NER) and a regex extractor run on every chunk simultaneously
— not one-or-the-other. The interesting engineering problem here was
**entity normalization**: GLiNER might extract "Pump P-101" while regex
extracts "P-101" for the same real-world asset, creating duplicate graph
nodes if left unreconciled. The fix (extract the embedded tag pattern from
within GLiNER's longer span before uppercasing) is a small function with an
outsized effect on graph quality — a lesson that **entity resolution matters
more than extraction recall** once two extractors are running.

## M2.5 — Industrial Ontology

**Goal:** give the graph a real is-a hierarchy, not a flat entity list.

24 entity labels, 15 relation types, explicit `ancestors_of`/`descendants_of`
functions — this is what lets a query reason at "show me all
RotatingEquipment incidents" altitude without enumerating Pump/Motor/
Compressor by hand. The hackathon problem statement's ontology sketch was a
flat list; building the hierarchy was an interpretive design decision, not a
transcription.

## M3 — Industrial Cognitive Engine

**Goal:** reasoning, not retrieval-and-paste.

The largest architectural decision in the project: an **8-stage LangGraph
pipeline** (Planner → Retrieval → Knowledge Graph → Evidence Fusion →
Reasoning → Recommendation → Verifier → Report) replacing an earlier simple
retrieve-then-answer flow. The key insight: **contradiction detection and
reasoning-graph construction need explicit structure**, not a long context
window handed to an LLM and hoped-for correct inference. Evidence Fusion
builds a per-query temporary graph connecting evidence items that share an
entity; the Reasoning stage and Contradiction Detector operate on that
structure deterministically, independent of which LLM (or none, in mock mode)
handles final narration. This is why contradiction detection works
identically whether `LLM_PROVIDER=mock` or `anthropic` — only the last stage
touches the LLM.

**Verifier ordering** was a deliberate deviation from a literal top-to-bottom
reading of an early agent list: it runs after Recommendation, not before,
because verification only makes sense once there's something to verify.

## M4 — Industrial Learning & Prediction Engine

**Goal:** rule-based prediction as the foundation later ML would replace.

The defining decision: **`PredictionEngine.register_model()`** as a runtime
swap point. Every prediction task type maps to a `PredictionModel` protocol
implementation; `RuleBasedPredictionModel` fills it today, but the interface
was designed so a trained model could replace it without changing
`PredictionEngine` itself — a bet that paid off directly in M6, two
milestones later, with zero modification to this file.

Also decided here: **M4 is independent of the M3 orchestrator.** This was an
explicit user choice at the time (documented in project history) and it set
the pattern every subsequent milestone followed — new capability gets its own
routes and `app.state` services, never edits into the highest-risk existing
code.

## M5 — AI Research, Benchmarking & Experimentation Framework

**Goal:** move from "it seems to work" to "here are the numbers, and they're
reproducible."

Every benchmark runner calls the *real* production components — the same
`EntityExtractionService`, `HybridRetriever`, etc. `main.py` wires up — never
a parallel test-only implementation. The A/B comparator's paired t-test uses
a normal approximation via `math.erf` rather than pulling in scipy for one
function — a deliberate dependency-minimalism choice. One genuine bug found
and fixed here: the t-test originally returned "no evidence" (p=1.0) when
paired differences had zero variance and a non-zero mean — actually the
*strongest* possible signal (every paired sample agreed exactly), not the
absence of one. Caught by a new test, not by inspection — a small reminder
that even "obviously correct" statistics code benefits from an adversarial
test case.

## M6 — Industrial Machine Learning Laboratory

**Goal:** real trained ML, without disturbing the deterministic reasoning
that came before.

Scikit-learn only (not XGBoost/LightGBM/PyTorch) was a deliberate scope
choice — breadth was traded for depth: every one of the seven ML subsystems
(dataset builder, feature store, training, registry, inference,
explainability, monitoring) is fully real, rather than several algorithm
backends each partially wired up. The `Trainer` protocol is algorithm-
agnostic by design, so this is a scope choice, not an architectural
limitation — an XGBoost trainer is a new class, not a rewrite.

**Two enums, not one:** `MLTaskType` (training-side) and `PredictionTaskType`
(serving-side, from M4) overlap but were kept separate rather than merged,
because "what was this model trained to predict" and "which live prediction
endpoint does it serve" are genuinely different concerns — conflating them
would have made the mapping between a training run and a deployed model
implicit instead of an explicit, auditable dict.

**Data leakage, caught before it shipped:** `SklearnTrainer.train()` did its
own internal train/test split but didn't originally expose the test set to
callers — meaning downstream diagnostics (confusion matrix, ROC, calibration
curve) computed by the API layer would have had to re-evaluate on the *full*
dataset, silently inflating every reported metric. Fixed by threading
`X_test`/`y_test` through `TrainedModel` before it became a bug report — this
is the kind of issue synthetic benchmarks with clean data can hide for a long
time if nobody thinks to check.

## RC1 — Enterprise Productization & Open Source Release

**Goal:** turn engineering excellence into a product a stranger can evaluate
without a guided tour.

The most interesting finding of this milestone wasn't a code bug — it was
what **running the Research Framework's own benchmarks against synthetic
data honestly revealed**: relation extraction scores 0 F1 on the synthetic
benchmark corpus (a benchmark-methodology gap, not proof the extractor is
broken — the hand-written `demo_data/` scenario produces real non-zero
relations through the identical code path), and the entity-extraction A/B
test is a regex-vs-regex tie in this development environment specifically
because GLiNER's model download isn't available here. Documenting *why* a
benchmark result looks weak, instead of omitting the benchmark or presenting
the number without context, is the same "never fabricate" principle that's
governed every milestone — applied to the reporting layer itself, not just
the application logic.

---

## Lessons learned

1. **Build the degradation path first.** Every fallback (`InMemoryGraphRepository`,
   `HashingEmbedder`, `PassthroughReranker`, `MockLLMProvider`) was designed
   in from the start, not retrofitted — this is why the entire platform runs
   with zero external services and no API key.
2. **Two extractors agreeing is more valuable than one extractor being
   clever.** GLiNER + regex, reconciled by a normalizer, beat either alone —
   and the reconciliation logic (not the extraction itself) was where the
   real engineering problem lived.
3. **A runtime swap point designed two milestones early pays off exactly
   once, exactly when needed.** `register_model()` (M4) was built with no
   concrete ML model to plug into it yet; M6 used it verbatim.
4. **Synthetic benchmarks can hide real gaps as easily as they can reveal
   them.** The relation-extraction 0 F1 result only surfaced because the
   benchmark was actually run and reported honestly, not assumed.
5. **The riskiest code benefits most from never being touched again.** The
   M3 orchestrator is the most complex part of the platform; every later
   milestone's discipline about not modifying it is *itself* validation that
   the layering was designed correctly.

## Architectural decisions (index)

| Decision | Milestone | Why |
| --- | --- | --- |
| Fallback-first infrastructure | M0 | Demo-ability without external services |
| Dual entity extraction + normalizer | M2 | Recall + precision without one model doing both |
| Explicit ontology hierarchy | M2.5 | Reasoning at the right altitude |
| 8-stage LangGraph pipeline over simple RAG | M3 | Structured evidence enables deterministic contradiction detection |
| Independent-subsystem rule | M4 | Never risk regressing the highest-stakes code |
| Runtime model registration (`register_model`) | M4 (used by M6) | Swap implementations without touching consumers |
| Benchmark against real components, not mocks | M5 | A benchmark measuring a stub measures nothing |
| Two separate task-type enums (training vs. serving) | M6 | Explicit, auditable mapping over implicit overloading |
| Held-out test set exposed on `TrainedModel` | M6 | Prevent data leakage into downstream diagnostics |
| Report benchmark limitations, not just numbers | RC1 | Same "never fabricate" principle applied to reporting |

## Tradeoffs accepted

- **SQLite over Postgres** for four separate stores — simple, zero-ops for a
  demo; a real production deployment would consolidate onto Postgres (every
  repository is already behind a Protocol for exactly this swap).
- **Scikit-learn only, not XGBoost/LightGBM/PyTorch** — depth over breadth;
  the `Trainer` protocol doesn't prevent adding them later.
- **Synthetic-but-learnable training labels** over no ML at all, given real
  historical outcome data doesn't exist in this deployment — documented
  explicitly rather than presented as real.
- **No authentication** — single-tenant, trusted-environment assumption,
  appropriate for a hackathon/portfolio deployment, explicitly flagged as a
  gap for anything beyond that.
- **In-process synchronous ingestion**, no background job queue — fine at
  demo scale, a documented bottleneck at production document volume.

## Future research directions

- Closing the synthetic-benchmark relation-extraction gap identified in RC1 —
  either richer synthetic ground truth or a benchmark methodology that
  doesn't collapse each document to a single chunk.
- Real SHAP value attribution as the next step beyond scikit-learn-native
  explainability (`Explainer` is already shaped to receive it as a same-
  interface addition).
- Graph neural network approaches to the `GRAPH` `MLProblemType`, currently
  unimplemented by design (no honest scikit-learn default exists for it).
- Windowed (not just cumulative) concept-drift detection, once enough
  evaluation history accumulates in a real deployment to make a trend
  meaningful rather than a single current snapshot.

## Potential publications

- **"Structured Evidence Fusion for Deterministic Contradiction Detection in
  LLM-Assisted Industrial QA"** — the M3 evidence-graph-before-reasoning
  architecture as an alternative to context-window-only RAG for domains where
  auditable, non-hallucinated conflict detection matters.
- **"Runtime-Swappable Prediction Interfaces: A Case Study in Rule-to-ML
  Migration Without Service Disruption"** — the M4→M6 `register_model()`
  pattern as a general technique for introducing ML into a rule-based system
  incrementally, one task type at a time, with a rule-based fallback always
  available.
- **"Honest Benchmarking: Reporting Synthetic Evaluation Gaps as Findings,
  Not Failures"** — the RC1 relation-extraction and GLiNER-availability
  findings as a methodological argument for publishing benchmark limitations
  alongside benchmark numbers, particularly relevant to reproducibility
  discussions in applied ML research.
