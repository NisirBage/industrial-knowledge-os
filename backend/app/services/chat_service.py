"""Chat service: runs a query through the full 8-stage agent orchestration
graph (planner -> retrieval -> knowledge_graph -> evidence_fusion ->
reasoning -> recommendation -> verifier -> report) and shapes the result
into the API response schema."""

import dataclasses
import time

from app.schemas.chat import ChatResponse, CitationOut, ContradictionOut, EvidenceOut, RecommendationOut


class ChatService:
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator

    def answer(self, query: str) -> ChatResponse:
        start = time.perf_counter()
        final_state = self._orchestrator.invoke({"query": query})
        overall_latency = round(time.perf_counter() - start, 4)

        latencies = dict(final_state.get("latencies", {}))
        latencies["overall"] = overall_latency

        reasoning_graph = final_state.get("reasoning_graph")
        recommendation_bundle = final_state.get("recommendation_bundle")

        return ChatResponse(
            summary=final_state.get("answer", ""),
            confidence=final_state.get("confidence", 0.0),
            citations=[CitationOut(**c) for c in final_state.get("citations", [])],
            evidence=[EvidenceOut(**dataclasses.asdict(e)) for e in final_state.get("evidence", [])],
            recommended_actions=final_state.get("recommended_actions", []),
            reasoning_summary=reasoning_graph.summary if reasoning_graph else "",
            contradictions=[ContradictionOut(**dataclasses.asdict(c)) for c in final_state.get("contradictions", [])],
            recommendations=[
                RecommendationOut(**dataclasses.asdict(r)) for r in (recommendation_bundle.recommendations if recommendation_bundle else [])
            ],
            latencies=latencies,
        )
