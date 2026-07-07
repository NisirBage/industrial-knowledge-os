from app.narration.narrator import LLMNarrator
from app.reasoning.graph_models import ReasoningGraph, ReasoningNode, ReasoningNodeType
from app.recommendation.models import RecommendationBundle


class _CapturingLLMProvider:
    def __init__(self):
        self.last_system_prompt = ""
        self.last_user_prompt = ""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        return "final answer"


def test_narrate_prompt_uses_structured_summary_not_raw_evidence_dump():
    provider = _CapturingLLMProvider()
    graph = ReasoningGraph(nodes=[], edges=[], summary="Analyzed 2 piece(s) of evidence.")
    graph.add_node(ReasoningNode(id="fact:c1", node_type=ReasoningNodeType.FACT, description="Pump P-101 needs lubrication", confidence=0.7))

    result = LLMNarrator(provider).narrate("How often is P-101 lubricated?", graph, RecommendationBundle(recommendations=[], overall_confidence=0.5))

    assert result == "final answer"
    assert "Analyzed 2 piece(s) of evidence." in provider.last_user_prompt
    assert "Pump P-101 needs lubrication" in provider.last_user_prompt


def test_narrate_includes_recommendations_when_present():
    from app.core.enums import Priority, Severity
    from app.recommendation.models import Recommendation

    provider = _CapturingLLMProvider()
    graph = ReasoningGraph(nodes=[], edges=[], summary="summary")
    recommendation = Recommendation(
        id="r1",
        description="Investigate bearing wear",
        priority=Priority.URGENT,
        severity=Severity.HIGH,
        estimated_impact="downtime risk",
        required_personnel=["Technician"],
        required_equipment=[],
        required_documents=[],
        estimated_downtime="TBD",
        confidence=0.8,
        justification="derived from risk node",
        supporting_evidence=["g1"],
    )

    LLMNarrator(provider).narrate("query", graph, RecommendationBundle(recommendations=[recommendation], overall_confidence=0.8))

    assert "Investigate bearing wear" in provider.last_user_prompt
    assert "urgent" in provider.last_user_prompt


def test_narrate_notes_absence_of_recommendations():
    provider = _CapturingLLMProvider()
    graph = ReasoningGraph(nodes=[], edges=[], summary="summary")

    LLMNarrator(provider).narrate("query", graph, RecommendationBundle(recommendations=[], overall_confidence=0.5))

    assert "No recommendations were generated." in provider.last_user_prompt
