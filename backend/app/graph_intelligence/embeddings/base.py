"""Graph Embedding Framework interfaces (M9 Component 3). Every
implementation (Node2Vec, DeepWalk, GCN, GraphSAGE, GAT, a future
heterogeneous GNN) satisfies `NodeEmbeddingModel` — the abstraction layer
never imports `torch_geometric` or `gensim` directly, only the concrete
adapter modules do, so nothing outside `embeddings/` is coupled to a
specific embedding library (the M9 spec's explicit "do not couple the
system to PyTorch Geometric").
"""

from typing import Protocol

from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.models import EmbeddingResult


class NodeEmbeddingModel(Protocol):
    name: str
    version: str

    def fit(self, dataset: GraphDataset) -> EmbeddingResult: ...


class EdgeEmbeddingModel(Protocol):
    """Edge embeddings are derived from a pair of node embeddings via an
    operator (Hadamard product, by default) rather than trained
    independently — the same choice the original Node2Vec paper's link-
    prediction experiments make, and the one Component 5 (Link Prediction)
    consumes directly."""

    def embed_edge(self, source_vector: list[float], target_vector: list[float]) -> list[float]: ...


class GraphEmbeddingModel(Protocol):
    """A single fixed-length vector for a set of nodes (e.g. a community or
    connected component) — mean-pooling by default. This is what Component
    4's graph-classification reinterpretation (classifying a cluster's
    aggregate risk, since there is one knowledge graph, not a dataset of
    many graphs) embeds."""

    def embed_graph(self, node_vectors: list[list[float]]) -> list[float]: ...
