from app.research.model_registry import (
    ComponentKind,
    entity_extractor_version,
    prediction_version,
    reasoner_version,
    recommendation_version,
    relation_extractor_version,
    retriever_version,
)


def test_retriever_version_has_correct_kind():
    version = retriever_version("hybrid", "v1", top_k="8")
    assert version.kind == ComponentKind.RETRIEVER
    assert version.name == "hybrid"
    assert version.version == "v1"
    assert version.config == {"top_k": "8"}


def test_entity_extractor_version_has_correct_kind():
    assert entity_extractor_version("gliner", "v1").kind == ComponentKind.ENTITY_EXTRACTOR


def test_relation_extractor_version_has_correct_kind():
    assert relation_extractor_version("rule_based", "v1").kind == ComponentKind.RELATION_EXTRACTOR


def test_reasoner_version_has_correct_kind():
    assert reasoner_version("reasoning_graph_builder", "v1").kind == ComponentKind.REASONER


def test_recommendation_version_has_correct_kind():
    assert recommendation_version("rule_based", "v1").kind == ComponentKind.RECOMMENDATION


def test_prediction_version_has_correct_kind():
    assert prediction_version("rule_based_v1", "v1").kind == ComponentKind.PREDICTION


def test_component_version_is_frozen_and_comparable():
    a = retriever_version("hybrid", "v1")
    b = retriever_version("hybrid", "v1")
    assert a == b
