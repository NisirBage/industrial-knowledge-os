# IKIOS v1.3.0 Release Notes

**Tag:** `v1.3.0` · **Date:** 2026-07-09 · **Codename:** Graph Intelligence

## What is IKIOS?

IKIOS (Industrial Knowledge Intelligence OS) is an industrial AI platform
built for the ET AI Hackathon 2026 (Problem Statement 8: "AI for Industrial
Knowledge Intelligence: Unified Asset & Operations Brain"). It ingests
industrial documents *and images*, builds a structured knowledge graph
spanning both, now learns from that graph via representation learning,
reasons over evidence with explicit contradiction detection, generates
justified recommendations, learns from feedback, trains real scikit-learn
and graph-embedding models, and exposes a full research/benchmarking
framework.

## Why this release exists

Every prior milestone read the Knowledge Graph via traversal and explicit
rules. This release adds a graph representation learning layer underneath
it — node embeddings, link prediction, node classification, and risk
propagation — without touching the deterministic Knowledge Graph
architecture or the symbolic Reasoning/Prediction pipeline those milestones
built. Graph Intelligence becomes another reasoning source; it never
replaces symbolic reasoning.

## What's new

- **`app/graph_intelligence/`** — a new backend module with ten
  sub-components: dataset builder (PyTorch Geometric, NetworkX, CSV,
  GraphML, JSON exports, deterministic ordering), feature store (structural
  + domain features, versioned like M6's Feature Store), an embedding
  framework (Node2Vec, DeepWalk, GCN, GraphSAGE, GAT behind shared
  Protocols — no `torch_geometric` coupling outside the module), link
  prediction, node classification (trained against the real rule-based
  model's own output — no fabricated ground truth), risk propagation (5
  types, merges into the M3 Industrial Reasoning Graph), and an
  explanation service.
- **Graph Intelligence API** (`/api/v1/graph-intelligence/*`) — summary,
  full graph export, per-node features, embedding training, similarity
  search, explanations, link-prediction candidates, node-classification
  training, and risk propagation simulation. See
  [docs/api/README.md](docs/api/README.md#graph-intelligence-m9).
- **Graph Intelligence Workspace** (`frontend/src/pages/
  GraphIntelligencePage.tsx`) — a sixth frontend tab with an interactive
  Cytoscape.js knowledge graph (zoom, filter by node type/name, community
  coloring), embedding training, similarity search with human-readable
  explanations, link prediction, risk propagation simulation, node
  classification training with a live model registry table (tagged Graph
  vs. Classical), and on-demand research benchmarks.
- **7 new research benchmarks** covering every embedding model plus link
  prediction and node classification — including a real train/test split
  for link prediction (an earlier version's exact-zero MRR turned out to be
  a genuine evaluation bug, not model weakness — fixed and documented).
- **No parallel infrastructure, literally**: every graph model registers
  into, is monitored by, and is explained through the exact same
  `SqlModelRegistry`/`ModelMonitor`/`Explainer` M6 built — confirmed by
  wiring real feature-importance output into the node-classification API
  response using the identical `Explainer` classical models use.

## Verified live, not just unit-tested

Every workflow — seeding the graph via real document upload, training an
embedding, running similarity search, training a node classifier and
watching it appear in the shared model registry tagged "Graph," running
every research benchmark — was exercised against the real running backend
and the real browser-rendered frontend during development, not just green
test output. One genuine frontend bug (an unconditional model-registry
refresh crashing on an undefined response) was caught this way and fixed
before release.

## Numbers

- **659 tests pass** (552 backend + 107 frontend, up from 523 backend / 66
  frontend), zero regressions in either. Spec required beyond 600.
- Real, unedited benchmark results are published in
  [docs/benchmarks/README.md](docs/benchmarks/README.md#graph-intelligence-benchmarks-m9),
  including one the model didn't "win": a simple rule baseline currently
  beats the graph-embedding node classifier on the small synthetic
  benchmark graph — reported as-is, not tuned away.

## Not changed

The Knowledge Graph schema, the M3 chat orchestrator, and every existing
API route's request/response shape are untouched. Graph Intelligence reads
from the same `GraphRepository` every prior milestone already built and
writes only new, additive registry entries and reasoning-graph nodes.

## Known limitations, stated plainly

- Heterogeneous (relation-type-aware) GNN layers are not implemented —
  only homogeneous GCN/GraphSAGE/GAT. `to_pyg_heterogeneous()` already
  produces the shape a future layer would consume.
- DGL export and real temporal graph learning are architecture-only (the
  `timestamp` field exists on graph nodes/edges; nothing consumes it yet).
- Node classification targets are the rule-based model's own output, not
  independently-verified historical ground truth (no such ground truth
  exists yet for this deployment) — an honest, explicit choice, not a
  hidden gap.
- The Graph Intelligence workspace's node selection does not yet sync
  across the Vision, Documents, or Chat workspaces.
- See [docs/roadmap/README.md](docs/roadmap/README.md#graph-intelligence-m9)
  for the complete, itemized list.

## Full history

See [CHANGELOG.md](CHANGELOG.md) for the structured per-version changelog
and [docs/architecture/graph-intelligence.md](docs/architecture/graph-intelligence.md)
for the full Graph Intelligence architecture writeup.
