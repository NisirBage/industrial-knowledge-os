from app.graph_intelligence.embeddings.models import EmbeddingResult, NodeEmbedding
from app.graph_intelligence.similarity import GraphSimilarityService


def _result() -> EmbeddingResult:
    return EmbeddingResult(
        embeddings=[
            NodeEmbedding("a", [1.0, 0.0]),
            NodeEmbedding("b", [1.0, 0.0]),  # identical direction to a
            NodeEmbedding("c", [0.0, 1.0]),  # orthogonal to a
            NodeEmbedding("d", [-1.0, 0.0]),  # opposite to a
        ],
        model_name="test",
        version="1.0",
        dimensions=2,
        training_seconds=0.0,
    )


def test_nearest_neighbors_ranks_most_similar_first():
    service = GraphSimilarityService()

    neighbors = service.nearest_neighbors(_result(), "a", k=3)

    assert neighbors[0][0] == "b"
    assert neighbors[0][1] == 1.0
    assert neighbors[-1][0] == "d"
    assert neighbors[-1][1] == -1.0


def test_nearest_neighbors_excludes_the_query_node_itself():
    service = GraphSimilarityService()

    neighbors = service.nearest_neighbors(_result(), "a", k=10)

    assert "a" not in [n[0] for n in neighbors]


def test_nearest_neighbors_respects_k():
    service = GraphSimilarityService()

    neighbors = service.nearest_neighbors(_result(), "a", k=1)

    assert len(neighbors) == 1


def test_nearest_neighbors_returns_empty_for_unknown_node():
    service = GraphSimilarityService()

    assert service.nearest_neighbors(_result(), "does-not-exist") == []
