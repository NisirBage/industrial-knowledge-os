"""Recommendation agent: turns RISK nodes and Contradictions from the
Reasoning Graph into traceable Recommendations, then attaches them back onto
the Reasoning Graph so the final narration can reference the full
fact -> cause -> effect -> risk -> recommendation chain."""

from app.orchestrator.state import AgentState
from app.recommendation.engine import RecommendationEngine


class RecommendationAgent:
    name = "recommendation"

    def __init__(self, engine: RecommendationEngine):
        self._engine = engine

    def run(self, state: AgentState) -> AgentState:
        reasoning_graph = state["reasoning_graph"]
        bundle = self._engine.generate(reasoning_graph, state.get("contradictions", []))
        self._engine.attach_to_reasoning_graph(reasoning_graph, bundle)
        return {**state, "reasoning_graph": reasoning_graph, "recommendation_bundle": bundle}
