import json

from app.contradiction.models import Contradiction
from app.core.enums import Priority, Severity
from app.reasoning.graph_models import ReasoningGraph, ReasoningNode, ReasoningNodeType
from app.recommendation.engine import RecommendationEngine


class _EchoLLMProvider:
    """Fails the JSON-refinement parse deliberately (mock-like), so the
    engine's fail-open fallback path is exercised."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "not json"


class _RefiningLLMProvider:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        num_lines = user_prompt.count("\n") + 1
        return json.dumps([f"refined {i}" for i in range(num_lines)])


def _risk_graph() -> ReasoningGraph:
    graph = ReasoningGraph(nodes=[], edges=[], summary="")
    graph.add_node(
        ReasoningNode(
            id="risk:g1",
            node_type=ReasoningNodeType.RISK,
            description="Recurring risk: P-101 FAILED_DUE_TO Bearing Wear",
            confidence=0.8,
            supporting_evidence_ids=["g1"],
        )
    )
    return graph


def test_risk_node_produces_recommendation_with_supporting_evidence():
    engine = RecommendationEngine(_EchoLLMProvider())
    bundle = engine.generate(_risk_graph(), [])

    assert len(bundle.recommendations) == 1
    rec = bundle.recommendations[0]
    assert rec.supporting_evidence == ["g1"]
    assert rec.priority == Priority.URGENT  # confidence 0.8 >= high threshold


def test_no_risk_no_contradiction_produces_no_recommendations():
    empty_graph = ReasoningGraph(nodes=[], edges=[], summary="")
    engine = RecommendationEngine(_EchoLLMProvider())

    bundle = engine.generate(empty_graph, [])

    assert bundle.recommendations == []


def test_contradiction_produces_recommendation_citing_conflicting_evidence():
    contradiction = Contradiction(
        conflicting_evidence=["c1", "c2"],
        confidence=0.6,
        severity=Severity.HIGH,
        resolution_strategy="verify",
        recommended_clarification="Confirm valve state",
    )
    engine = RecommendationEngine(_EchoLLMProvider())

    bundle = engine.generate(ReasoningGraph(nodes=[], edges=[], summary=""), [contradiction])

    assert len(bundle.recommendations) == 1
    assert bundle.recommendations[0].supporting_evidence == ["c1", "c2"]
    assert bundle.recommendations[0].priority == Priority.HIGH


def test_multiple_contradictions_on_the_same_evidence_pair_get_unique_recommendation_ids():
    contradiction = Contradiction(
        conflicting_evidence=["c1", "c2"],
        confidence=0.6,
        severity=Severity.HIGH,
        resolution_strategy="verify",
        recommended_clarification="Confirm valve state",
    )
    engine = RecommendationEngine(_EchoLLMProvider())

    bundle = engine.generate(ReasoningGraph(nodes=[], edges=[], summary=""), [contradiction, contradiction])

    assert len(bundle.recommendations) == 2
    ids = {r.id for r in bundle.recommendations}
    assert len(ids) == 2


def test_refinement_failure_falls_back_to_rule_generated_justification():
    engine = RecommendationEngine(_EchoLLMProvider())
    bundle = engine.generate(_risk_graph(), [])

    assert "Derived from the reasoning graph" in bundle.recommendations[0].justification


def test_refinement_success_replaces_justification_text():
    engine = RecommendationEngine(_RefiningLLMProvider())
    bundle = engine.generate(_risk_graph(), [])

    assert bundle.recommendations[0].justification == "refined 0"


def test_attach_to_reasoning_graph_adds_recommendation_node_and_edge():
    graph = _risk_graph()
    graph.add_node(ReasoningNode(id="fact:g1", node_type=ReasoningNodeType.FACT, description="fact", confidence=0.8, supporting_evidence_ids=["g1"]))
    engine = RecommendationEngine(_EchoLLMProvider())
    bundle = engine.generate(graph, [])

    engine.attach_to_reasoning_graph(graph, bundle)

    recommendation_nodes = graph.nodes_of_type(ReasoningNodeType.RECOMMENDATION)
    assert len(recommendation_nodes) == 1
    assert any(e.relationship == "informs" and e.source_id == "fact:g1" for e in graph.edges)


def test_overall_confidence_is_mean_of_recommendation_confidences():
    engine = RecommendationEngine(_EchoLLMProvider())
    bundle = engine.generate(_risk_graph(), [])
    assert bundle.overall_confidence == 0.8
