from app.graph_intelligence.embeddings.models import EmbeddingResult, NodeEmbedding
from app.graph_intelligence.explanation import GraphExplanationService
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def _seeded_repo():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-201")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.6)
    return repo, pump_id, valve_id


def _embedding_result(pump_id, valve_id):
    return EmbeddingResult(
        embeddings=[NodeEmbedding(pump_id, [1.0, 0.0]), NodeEmbedding(valve_id, [0.9, 0.1])],
        model_name="test", version="1.0", dimensions=2, training_seconds=0.0,
    )


def test_explain_node_includes_nearest_neighbors():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.8)

    assert explanation.nearest_neighbors[0][0] == valve_id


def test_explain_node_includes_important_edges_from_the_graph():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.8)

    assert len(explanation.important_edges) == 1
    assert explanation.important_edges[0]["neighbor_canonical_name"] == "V-201"


def test_explain_node_includes_supporting_evidence():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.8)

    assert len(explanation.supporting_evidence) == 1
    assert explanation.supporting_evidence[0]["canonical_name"] == "V-201"


def test_explain_node_includes_community_id_when_structural_features_supplied():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(
        pump_id, _embedding_result(pump_id, valve_id), confidence=0.8, structural_features={pump_id: {"community_id": 3}}
    )

    assert explanation.community_id == 3


def test_explain_node_community_is_none_without_structural_features():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.8)

    assert explanation.community_id is None


def test_confidence_is_carried_through_to_the_explanation():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)

    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.42)

    assert explanation.confidence == 0.42


def test_to_text_produces_a_human_readable_summary():
    repo, pump_id, valve_id = _seeded_repo()
    service = GraphExplanationService(repo)
    explanation = service.explain_node(pump_id, _embedding_result(pump_id, valve_id), confidence=0.8)

    text = GraphExplanationService.to_text(explanation)

    assert "Nearest embedding neighbors" in text
    assert "Confidence: 80.0%" in text
