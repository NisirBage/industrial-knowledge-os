# Unified Execution Trace & Failure Intelligence (O3.3 / O3.4 / O3.4.1)

Every chat turn now produces one canonical `ExecutionTrace` — the single
source of truth for runtime analysis. Nothing about the orchestrator's
stages changed; a wrapper around each stage (`_timed` in
`app/orchestrator/graph.py`) additionally emits immutable
`ExecutionEvent`s alongside the existing `latencies` dict. Failure Pattern
Mining, Root Cause Graphs, Counterfactual Simulation, and Optimization
Recommendations are all built exclusively on top of `ExecutionTrace` —
none of them re-derive telemetry independently, and none of them touch
production settings.

```mermaid
flowchart TD
    ORCH[Orchestrator stages] -->|_timed wrapper| EVT[ExecutionEvent Started/Finished]
    EVT --> ER[ExecutionRecorder]
    ER --> ET[(ExecutionTrace)]
    ET --> REPO[(SqlExecutionRepository)]
    ET --> MINER[FailurePatternMiner]
    ET --> CLUSTER[Clustering Strategies]
    ET --> RCG[RootCauseGraphBuilder]
    ET --> CF[CounterfactualSimulator]
    MINER --> PATTERNS[(SqlFailurePatternRepository)]
    PATTERNS --> OPT[OptimizationRecommendationEngine]
    CF --> OPT
    REPO --> API[/api/v1/executions/*]
    PATTERNS --> API2[/api/v1/failure-intelligence/*]
    API --> UI1[Execution Explorer]
    API2 --> UI2[Failure Intelligence workspace]
```

## O3.3 — ExecutionTrace architecture

**`ExecutionTrace` / `ExecutionEvent`**
([`app/execution/models.py`](../../backend/app/execution/models.py)): the
canonical model. An `ExecutionTrace` carries `execution_id`, `query`, its
list of `events`, `latencies`, `memory`, final `confidence`, `timestamps`,
and 14 per-subsystem trace slots (`planner_trace` … `cache_trace`). Each
slot is a small real-signal dict pulled from that stage's own
already-computed output — never a duplicate of the full domain object.
Subsystems not invoked inside live chat (Compression, Vision, Prediction,
Graph Intelligence) simply leave their slot `None`.

**`ExecutionRecorder`** ([`recorder.py`](../../backend/app/execution/recorder.py)):
builds Started/Finished event pairs with real `psutil` memory deltas, and
assembles the final trace from an orchestrator run's accumulated events.
No hidden global state — every method takes what it needs and returns a
new value.

**Pipeline instrumentation**: `app/orchestrator/graph.py`'s `_timed`
wrapper (already the observability mechanism for `latencies`) now also
appends events into a new, additive `AgentState["execution_events"]` field.
No agent's own code is aware this exists.

**`SqlExecutionRepository`** ([`repository.py`](../../backend/app/execution/repository.py)):
persists traces, events, and `FailureEvent`s in dedicated tables (same
pattern as `SqlExperimentTracker`). `ChatService` optionally persists a
trace per chat turn — persistence failures are swallowed so they can never
break the chat response (see `_persist_execution_trace`).

**Timeline Reconstruction** ([`timeline.py`](../../backend/app/execution/timeline.py)):
given one trace, reconstructs ordered spans, component ordering, real
overlap-detected parallel branches, confidence evolution, and a DAG. The
current orchestrator runs strictly sequentially, so overlap detection
honestly reports no parallel branches today — it becomes meaningful the
moment a future stage genuinely runs concurrently.

**Migration adapters** (O3.3 Component 10): `ErrorAnalyzer.analyze_trace()`
and `SelfReflectionEngine.reflect_trace()` consume `ExecutionTrace`
directly instead of the original domain objects, proving the pattern for
future subsystems — the original `analyze()`/`reflect()` signatures are
unchanged.

## O3.4 — Failure Pattern Mining & Root Cause Intelligence

**`FailurePatternMiner`** ([`miner.py`](../../backend/app/failure_intelligence/miner.py)):
runs `ErrorAnalyzer.analyze_trace()` across a batch of traces and
aggregates matching categories into `FailurePattern`s (support, frequency,
severity, affected components, example execution ids). Also mines a
cross-stage **confidence collapse** signal (final confidence far below
retrieval confidence) and **repeated failure sequence** patterns (shared
component ordering across failing executions). Persisted via
`SqlFailurePatternRepository`.

**Clustering** ([`clustering.py`](../../backend/app/failure_intelligence/clustering.py)):
pluggable `ClusteringStrategy` protocol — `RuleBasedClusteringStrategy`
(category / component sequence / query intent / entity / timeline
proximity), `EmbeddingBasedClusteringStrategy` (real `Embedder`, cosine
similarity), `GraphBasedClusteringStrategy` (real `GraphRepository`, shared
Knowledge Graph nodes).

**Root Cause Graph** ([`root_cause_graph.py`](../../backend/app/failure_intelligence/root_cause_graph.py)):
7 node types (Failure/Execution/Claim/Recommendation/Component/Evidence/
Pattern), 5 edge types (`CAUSED_BY`/`CONTRIBUTED_TO`/`CORRELATED_WITH`/
`PREVENTED_BY`/`AMPLIFIED_BY`). Supports traversal, dominant-cause lookup,
dependency-depth, critical-path (BFS), and export. `AMPLIFIED_BY` edges
are an explicitly documented correlation heuristic (upstream
retrieval/ranking weakness coinciding with downstream verification
problems) — never asserted as proven causality.

**`FailureAttributionEngine`** ([`attribution.py`](../../backend/app/failure_intelligence/attribution.py)):
primary/secondary cause + contributing components + evidence/execution
confidence, computed only from a pattern's own real fields. Reports
`"unattributed"` rather than fabricating a cause when a pattern has no
affected components.

**`CounterfactualSimulator`** ([`counterfactual.py`](../../backend/app/failure_intelligence/counterfactual.py)):
8 toggles (planner/retriever/ranker/compression/verification/nli/
recommendation_verification/faithfulness). Applies a documented
deterministic transform to the relevant trace field, then re-runs
`analyze_trace()` on both the original and modified trace to see which
failure categories disappear. For check-disabling toggles
(verification/NLI/faithfulness/recommendation verification), the result's
`note` explicitly states the alarm — not necessarily the underlying issue —
disappeared.

**`OptimizationRecommendationEngine`** ([`optimization.py`](../../backend/app/failure_intelligence/optimization.py)):
deterministic, template-driven recommendations generated only from real
`FailurePattern`s / `CounterfactualResult`s passed in — every
recommendation cites its supporting evidence and execution ids. Advisory
only; nothing here modifies production settings or retrains a model.

## O3.4.1 — Execution Intelligence frontend workspace

Two new Research Dashboard tabs consume the endpoints above exclusively —
no new backend logic was added to support them, and no duplicated API
calls or graph-rendering logic exist across pages:

- **Executions** ([`ExecutionsPage.tsx`](../../frontend/src/pages/ExecutionsPage.tsx)):
  search/sort/paginate/filter (query, component, date range, "has a
  detected failure pattern") over `GET /executions`; a detail inspector
  per execution renders a Pipeline Waterfall (recharts stacked bars over
  real span offsets/durations), a confidence-evolution line chart, and an
  Execution DAG.
- **Failure Intelligence** ([`FailureIntelligencePage.tsx`](../../frontend/src/pages/FailureIntelligencePage.tsx)):
  Dashboard Metrics cards (execution/failure/pattern counts, failure rate,
  average confidence/latency, cache hit ratio — the last reusing the real
  `execution_trace_overhead` research benchmark rather than inventing a
  new endpoint), a Failure Pattern Explorer with drill-down into
  supporting executions, a Root Cause Graph viewer (dominant
  cause/dependency depth/critical path computed client-side over the
  fetched graph, JSON export), a Counterfactual Explorer, and an
  Optimization Recommendation viewer.

**`CytoscapeGraph`** ([`components/CytoscapeGraph.tsx`](../../frontend/src/components/CytoscapeGraph.tsx)):
the cytoscape lifecycle (init/destroy/zoom/fit/selection-highlighting) was
extracted out of `GraphIntelligencePage` into a shared component — it is
now the single graph-rendering mechanism reused by Graph Intelligence, the
Execution DAG viewer, and the Root Cause Graph viewer. Domain-specific
node/edge construction and stylesheets stay with each caller.

**Honestly out of scope**: the execution/failure-intelligence API
endpoints are read-side and on-demand-compute only; there is no live
"streaming" trace viewer, and pattern mining/clustering run against
whatever traces are currently persisted rather than a continuously
updating background job.
