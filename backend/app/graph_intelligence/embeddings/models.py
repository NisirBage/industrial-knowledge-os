"""Graph Embedding Framework data model (M9 Component 3)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeEmbedding:
    node_id: str
    vector: list[float]


@dataclass(frozen=True)
class EmbeddingResult:
    embeddings: list[NodeEmbedding]
    model_name: str
    version: str
    dimensions: int
    training_seconds: float

    def vector_for(self, node_id: str) -> list[float] | None:
        return next((e.vector for e in self.embeddings if e.node_id == node_id), None)
