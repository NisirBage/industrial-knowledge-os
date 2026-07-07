"""Report agent: the final stage. Calls the LLM Narrator (the LLM's only job
in the pipeline) and shapes the backward-compatible ChatResponse fields
(citations, recommended_actions, confidence) from the structured bundle/
reasoning graph/recommendations — never from a fresh LLM call.
"""

from app.narration.narrator import LLMNarrator
from app.orchestrator.state import AgentState
from app.recommendation.models import RecommendationBundle


class ReportAgent:
    name = "report"

    def __init__(self, narrator: LLMNarrator):
        self._narrator = narrator

    def run(self, state: AgentState) -> AgentState:
        reasoning_graph = state["reasoning_graph"]
        recommendation_bundle = state.get("recommendation_bundle") or RecommendationBundle(recommendations=[], overall_confidence=0.5)
        bundle = state["evidence_bundle"]

        answer = self._narrator.narrate(state["query"], reasoning_graph, recommendation_bundle)

        citations = [
            {
                "index": i + 1,
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "page_number": item.page_number,
                "source": item.source,
                "entity_ids": item.entity_ids,
            }
            for i, item in enumerate(bundle.all_evidence())
        ]

        recommended_actions = [r.description for r in recommendation_bundle.recommendations]
        if not recommended_actions and not bundle.all_evidence():
            recommended_actions = ["Retrieve more source documents — no supporting evidence was found for this query."]

        confidence = round((bundle.confidence + recommendation_bundle.overall_confidence) / 2, 2) if recommendation_bundle.recommendations else bundle.confidence

        return {
            **state,
            "answer": answer,
            "confidence": confidence,
            "citations": citations,
            "recommended_actions": recommended_actions,
        }
