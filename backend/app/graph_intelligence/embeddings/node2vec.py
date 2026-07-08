"""Node2Vec / DeepWalk (M9 Component 3): real biased-random-walk + skip-gram
embeddings via gensim's Word2Vec — the same "walks are sentences, nodes are
words" formulation the original Node2Vec/DeepWalk papers use. No PyTorch
involved; this family is deliberately the lightweight, CPU-only embedding
option, distinct from the GNN family in gnn.py.
"""

import time

import networkx as nx
from gensim.models import Word2Vec

from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.models import EmbeddingResult, NodeEmbedding
from app.graph_intelligence.embeddings.random_walk import generate_walks

_DEFAULT_DIMENSIONS = 32
_DEFAULT_WINDOW = 5
_DEFAULT_EPOCHS = 5


class Node2VecEmbedding:
    """p < 1 or q < 1 (the Node2Vec-proper setting) biases the walk toward
    local/global exploration respectively; p=q=1.0 makes this identical to
    DeepWalk — see DeepWalkEmbedding below, which is a thin alias."""

    name = "node2vec"
    version = "1.0"

    def __init__(self, dimensions: int = _DEFAULT_DIMENSIONS, p: float = 1.0, q: float = 0.5, seed: int = 42):
        self._dimensions = dimensions
        self._p = p
        self._q = q
        self._seed = seed

    def fit(self, dataset: GraphDataset) -> EmbeddingResult:
        start = time.perf_counter()
        nx_graph = self._to_undirected_networkx(dataset)

        if nx_graph.number_of_nodes() == 0:
            return EmbeddingResult(embeddings=[], model_name=self.name, version=self.version, dimensions=self._dimensions, training_seconds=0.0)

        walks = generate_walks(nx_graph, p=self._p, q=self._q, seed=self._seed)
        model = Word2Vec(
            walks,
            vector_size=self._dimensions,
            window=_DEFAULT_WINDOW,
            min_count=0,
            sg=1,
            workers=1,
            epochs=_DEFAULT_EPOCHS,
            seed=self._seed,
        )

        embeddings = [
            NodeEmbedding(node_id=node.id, vector=model.wv[node.id].tolist() if node.id in model.wv else [0.0] * self._dimensions)
            for node in dataset.nodes
        ]
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.name,
            version=self.version,
            dimensions=self._dimensions,
            training_seconds=round(time.perf_counter() - start, 4),
        )

    @staticmethod
    def _to_undirected_networkx(dataset: GraphDataset) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(node.id for node in dataset.nodes)
        for edge in dataset.edges:
            graph.add_edge(edge.source, edge.target)
        return graph


class DeepWalkEmbedding(Node2VecEmbedding):
    """DeepWalk is Node2Vec with p=q=1.0 — an unbiased random walk. A thin
    subclass, not a reimplementation, because that is the actual
    relationship between the two algorithms."""

    name = "deepwalk"

    def __init__(self, dimensions: int = _DEFAULT_DIMENSIONS, seed: int = 42):
        super().__init__(dimensions=dimensions, p=1.0, q=1.0, seed=seed)
