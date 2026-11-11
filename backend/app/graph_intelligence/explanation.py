"""GraphExplanationService (M9 Graph Intelligence API): the single place
every graph prediction (Link Prediction, Node Classification, Risk
Propagation) gets its explanation from — nearest neighbors, important
edges, community, supporting evidence, confidence — so "graph models must
never become black boxes" is satisfied once, not re-implemented per
component.
"""

from dataclasses import dataclass

from app.graph_intelligence.embeddings.models import EmbeddingResult
from app.graph_intelligence.similarity import GraphSimilarityService
from app.repositories.graph_repository import GraphRepositoryProtocol


@dataclass(frozen=True)
class GraphExplanation:
    node_id: str
    nearest_neighbors: list[tuple[str, float]]
    important_edges: list[dict]
    community_id: int | None
    supporting_evidence: list[dict]
    confidence: float


class GraphExplanationService:
    def __init__(self, graph_repository: GraphRepositoryProtocol, similarity_service: GraphSimilarityService | None = None):
        self._graph_repository = graph_repository
        self._similarity_service = similarity_service or GraphSimilarityService()

    def explain_node(
        self,
        node_id: str,
        embedding_result: EmbeddingResult,
        confidence: float,
        structural_features: dict[str, dict] | None = None,
        top_k_neighbors: int = 5,
    ) -> GraphExplanation:
        nearest_neighbors = self._similarity_service.nearest_neighbors(embedding_result, node_id, k=top_k_neighbors)

        graph_neighbors = self._graph_repository.find_neighbors(node_id, depth=1)
        important_edges = [
            {"neighbor_id": n["id"], "neighbor_canonical_name": n["canonical_name"], "relation_type": n["relation_type"]}
            for n in graph_neighbors[:top_k_neighbors]
        ]
        supporting_evidence = [
            {"node_id": n["id"], "type": n["type"], "canonical_name": n["canonical_name"]} for n in graph_neighbors
        ]

        community_id = None
        if structural_features is not None:
            community_id = structural_features.get(node_id, {}).get("community_id")

        return GraphExplanation(
            node_id=node_id,
            nearest_neighbors=nearest_neighbors,
            important_edges=important_edges,
            community_id=community_id,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
        )

    @staticmethod
    def to_text(explanation: GraphExplanation) -> str:
        neighbor_summary = ", ".join(f"{nid} ({score})" for nid, score in explanation.nearest_neighbors[:3]) or "none found"
        edge_summary = ", ".join(f"{e['relation_type']}->{e['neighbor_canonical_name']}" for e in explanation.important_edges[:3]) or "none found"
        community = explanation.community_id if explanation.community_id is not None else "unknown"
        return (
            f"Nearest embedding neighbors: {neighbor_summary}. "
            f"Important graph edges: {edge_summary}. "
            f"Community: {community}. "
            f"Supporting evidence: {len(explanation.supporting_evidence)} connected graph node(s). "
            f"Confidence: {round(explanation.confidence * 100, 1)}%."
        )
