"""Concrete EdgeEmbeddingModel/GraphEmbeddingModel implementations (M9
Component 3) — both derived from node embeddings via a fixed operator
rather than trained independently, the standard approach for both (see
base.py's docstrings for why).
"""


class HadamardEdgeEmbedding:
    """Element-wise product of the two endpoint embeddings — the operator
    the original Node2Vec paper's link-prediction experiments found most
    effective, and the one Component 5 (Link Prediction) uses."""

    name = "hadamard"

    def embed_edge(self, source_vector: list[float], target_vector: list[float]) -> list[float]:
        return [s * t for s, t in zip(source_vector, target_vector, strict=True)]


class MeanPoolingGraphEmbedding:
    """Mean of every node vector in a set (a community, a connected
    component) — the graph-level embedding Component 4's graph-
    classification reinterpretation (classifying a cluster's aggregate
    risk) consumes."""

    name = "mean_pooling"

    def embed_graph(self, node_vectors: list[list[float]]) -> list[float]:
        if not node_vectors:
            return []
        dimensions = len(node_vectors[0])
        sums = [0.0] * dimensions
        for vector in node_vectors:
            for i, value in enumerate(vector):
                sums[i] += value
        return [s / len(node_vectors) for s in sums]
