"""GraphSimilarityService (M9 Graph Intelligence API): nearest-neighbor
search over any embedding family's output — Node2Vec, DeepWalk, GCN,
GraphSAGE, GAT all produce the same `EmbeddingResult` shape, so this one
service works with all of them, no per-model-family special casing.
"""

import math

from app.graph_intelligence.embeddings.models import EmbeddingResult


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class GraphSimilarityService:
    def nearest_neighbors(self, embedding_result: EmbeddingResult, node_id: str, k: int = 5) -> list[tuple[str, float]]:
        target_vector = embedding_result.vector_for(node_id)
        if target_vector is None:
            return []

        scored = [
            (embedding.node_id, round(_cosine_similarity(target_vector, embedding.vector), 4))
            for embedding in embedding_result.embeddings
            if embedding.node_id != node_id
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
