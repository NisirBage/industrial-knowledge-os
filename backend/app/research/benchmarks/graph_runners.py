"""Graph Intelligence benchmark runners (M9 Component 9) — extends the M5
Benchmark Suite exactly the way M7's vision_runners.py did: same
`BenchmarkReport` shape, same `ExperimentManager` integration, same
"generate a small deterministic synthetic graph, measure the real
production component" discipline. No parallel research framework.
"""

import random
import time
import tracemalloc

from app.evaluation.metrics import mean_absolute_error, mrr, recall_at_k
from app.evaluation.models import NamedMetric
from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.gnn import GATEmbedding, GCNEmbedding, GraphSAGEEmbedding
from app.graph_intelligence.embeddings.node2vec import DeepWalkEmbedding, Node2VecEmbedding
from app.graph_intelligence.link_prediction import LinkPredictionTrainer, LinkPredictor
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository
from app.research.benchmarks.models import BenchmarkReport

_EMBEDDING_MODEL_CLASSES = {
    "node2vec": Node2VecEmbedding,
    "deepwalk": DeepWalkEmbedding,
    "gcn": GCNEmbedding,
    "graphsage": GraphSAGEEmbedding,
    "gat": GATEmbedding,
}


_SYNTHETIC_NODE_TYPES = [EntityLabel.PUMP, EntityLabel.VALVE, EntityLabel.MOTOR]


def _synthetic_graph_repo(num_nodes: int = 20, seed: int = 42) -> InMemoryGraphRepository:
    """A deterministic ring-plus-shortcuts graph — enough real structure
    (cycles, varying degree) for centrality/embedding/link-prediction
    metrics to be meaningful, unlike a single chain. Node types are
    deliberately varied (not all Pump) — an earlier version used one type
    for every node, which made the Dataset Builder's default one-hot node
    feature degenerate (every row identical), silently starving the GNN
    embedding models of any real per-node input signal. Caught by an
    embedding_quality_margin sitting at exactly 0.0 for GCN/GraphSAGE/GAT
    during development — the real KG has many node types, so this fix also
    makes the benchmark more representative, not just less broken.
    """
    rng = random.Random(seed)
    repo = InMemoryGraphRepository()
    ids = [repo.create_node(_SYNTHETIC_NODE_TYPES[i % len(_SYNTHETIC_NODE_TYPES)], f"P-{i}") for i in range(num_nodes)]
    for i in range(num_nodes):
        repo.create_relation(ids[i], ids[(i + 1) % num_nodes], RelationType.CONNECTED_TO, confidence=0.6)
    for _ in range(num_nodes // 3):
        a, b = rng.sample(range(num_nodes), 2)
        repo.create_relation(ids[a], ids[b], RelationType.RELATED_TO, confidence=0.5)
    return repo


class GraphEmbeddingBenchmark:
    """Purpose: measure training time, memory footprint, and embedding
    quality for one embedding model family. Embedding quality is a real,
    label-free proxy — mean cosine similarity of directly-connected node
    pairs minus mean cosine similarity of random pairs; a model that
    captures graph structure should score positive (neighbors closer than
    strangers), not by construction, but as a genuine property of the
    learned vectors. Limitations: memory is measured as process RSS delta
    via `tracemalloc`, which captures Python-level allocations but not all
    native (C++/CUDA) allocations a framework might make internally.
    """

    def __init__(self, model_name: str, num_nodes: int = 20):
        if model_name not in _EMBEDDING_MODEL_CLASSES:
            raise ValueError(f"Unknown embedding model {model_name!r}. Available: {sorted(_EMBEDDING_MODEL_CLASSES)}")
        self.name = f"graph_embedding_{model_name}"
        self._model_name = model_name
        self._num_nodes = num_nodes

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        repo = _synthetic_graph_repo(self._num_nodes)
        dataset = GraphDatasetBuilder(repo).build()

        tracemalloc.start()
        model = _EMBEDDING_MODEL_CLASSES[self._model_name](dimensions=16)
        result = model.fit(dataset)
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        connected_pairs = {(e.source, e.target) for e in dataset.edges}
        node_ids = [n.id for n in dataset.nodes]
        rng = random.Random(7)
        random_pairs = [(rng.choice(node_ids), rng.choice(node_ids)) for _ in range(len(connected_pairs))]

        def _cosine(a_id: str, b_id: str) -> float | None:
            a, b = result.vector_for(a_id), result.vector_for(b_id)
            if a is None or b is None:
                return None
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

        connected_scores = [s for s in (_cosine(a, b) for a, b in connected_pairs) if s is not None]
        random_scores = [s for s in (_cosine(a, b) for a, b in random_pairs if a != b) if s is not None]
        mean_connected = round(sum(connected_scores) / len(connected_scores), 4) if connected_scores else 0.0
        mean_random = round(sum(random_scores) / len(random_scores), 4) if random_scores else 0.0

        metrics = [
            NamedMetric("training_seconds", result.training_seconds),
            NamedMetric("peak_memory_bytes", float(peak_memory_bytes)),
            NamedMetric("mean_connected_similarity", mean_connected),
            NamedMetric("mean_random_similarity", mean_random),
            NamedMetric("embedding_quality_margin", round(mean_connected - mean_random, 4)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ring_graph", len(dataset.nodes))


class LinkPredictionBenchmark:
    """Purpose: measure link-prediction quality by holding out real edges,
    then checking whether the trained model ranks the true target above
    randomly-sampled non-targets for the same source — MRR and Recall@5
    computed via the same shared `app.evaluation.metrics` functions
    HybridRetrievalBenchmark already uses, not a separate metric
    implementation.
    """

    name = "graph_link_prediction"

    def __init__(self, num_nodes: int = 20):
        self._num_nodes = num_nodes

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        repo = _synthetic_graph_repo(self._num_nodes)
        full_dataset = GraphDatasetBuilder(repo).build()

        # A real train/test split: held-out edges are physically removed
        # from the graph the embeddings and link predictor are trained on,
        # not just set aside for scoring — otherwise `top_candidates()`'s
        # (correct) exclusion of already-connected nodes would make the
        # true target structurally impossible to ever rank, regardless of
        # model quality. This was a genuine bug caught by a suspicious
        # exact-zero MRR/Recall@5 during development, not a hypothetical.
        held_out_edges = full_dataset.edges[: min(10, len(full_dataset.edges) // 2)]
        held_out_set = {(e.source, e.target) for e in held_out_edges}
        train_edges = [e for e in full_dataset.edges if (e.source, e.target) not in held_out_set]
        train_dataset = GraphDataset(nodes=full_dataset.nodes, edges=train_edges, node_id_to_index=full_dataset.node_id_to_index)

        embedding_result = Node2VecEmbedding(dimensions=16).fit(train_dataset)
        model = LinkPredictionTrainer().fit(embedding_result, train_dataset)
        predictor = LinkPredictor(repo)

        ranked_relevance: list[list[bool]] = []
        recalls: list[float] = []
        for edge in held_out_edges:
            candidates = predictor.top_candidates(edge.source, train_dataset, embedding_result, model, top_k=len(train_dataset.nodes))
            ranked_ids = [c.target_id for c in candidates]
            ranked_relevance.append([rid == edge.target for rid in ranked_ids])
            recalls.append(recall_at_k({edge.target}, ranked_ids, k=5))

        mrr_value = mrr(ranked_relevance)
        recall_value = round(sum(recalls) / len(recalls), 4) if recalls else 0.0

        metrics = [
            NamedMetric("mrr", mrr_value),
            NamedMetric("recall_at_5", recall_value),
            NamedMetric("held_out_edges", float(len(held_out_edges))),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ring_graph", len(held_out_edges))


class NodeClassificationBenchmark:
    """Purpose: compare the rule-based baseline against a graph-embedding
    model predicting the *same* real target (the rule model's own output —
    see node_classification.py's module docstring for why this is an
    honest comparison, not two models fit to invented numbers). Reports
    both models' mean absolute error against held-out equipment.
    """

    name = "graph_node_classification"

    def __init__(self, num_nodes: int = 20):
        self._num_nodes = num_nodes

    async def run(self) -> BenchmarkReport:
        import pandas as pd
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        from app.ml.datasets.models import MLProblemType
        from app.ml.training.trainer import SklearnTrainer

        start = time.perf_counter()
        repo = _synthetic_graph_repo(self._num_nodes)
        dataset = GraphDatasetBuilder(repo).build()
        embedding_result = Node2VecEmbedding(dimensions=16).fit(dataset)

        # Synthetic-but-documented target: degree-derived risk proxy (a real
        # graph-structural quantity, not a fabricated label) — used as a
        # regression target both a "rule" (degree-based formula) and a
        # graph-embedding regressor try to predict, matching the honest
        # "synthetic labels, real formula" precedent M6's dataset generator
        # already established.
        graph = GraphDatasetBuilder.to_networkx(dataset).to_undirected()
        degrees = dict(graph.degree())
        rows, targets = [], []
        for node in dataset.nodes:
            vector = embedding_result.vector_for(node.id)
            if vector is None:
                continue
            rows.append({f"dim_{i}": v for i, v in enumerate(vector)})
            targets.append(float(degrees.get(node.id, 0)))

        X = pd.DataFrame(rows)
        y = pd.Series(targets)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Rule baseline: predict the training mean — a real, deliberately
        # simple rule (not a fabricated number) to compare the graph model
        # against, in the same spirit as M4's rule-based prediction models.
        rule_predictions = [float(y_train.mean())] * len(X_test)
        rule_mae = mean_absolute_error(rule_predictions, list(y_test))

        trainer = SklearnTrainer(algorithm_factories={MLProblemType.GRAPH: lambda: RandomForestRegressor(n_estimators=100, random_state=42)})
        trained = trainer.train(X, y, MLProblemType.GRAPH, "graph_benchmark_task", test_size=0.3, random_state=42)
        graph_predictions = trained.estimator.predict(X_test).tolist()
        graph_mae = mean_absolute_error(graph_predictions, list(y_test))

        metrics = [
            NamedMetric("rule_baseline_mae", rule_mae),
            NamedMetric("graph_model_mae", graph_mae),
            NamedMetric("graph_model_wins", float(graph_mae < rule_mae)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ring_graph", len(X_test))
