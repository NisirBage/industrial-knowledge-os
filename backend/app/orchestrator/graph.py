"""LangGraph wiring: planner -> retrieval -> knowledge_graph -> evidence_fusion
-> reasoning -> recommendation -> verifier -> report.

Every node is wrapped with a latency timer (Observability requirement) that
records elapsed seconds into state["latencies"][stage_name], so per-stage
cost is measurable without touching each agent's own code.

Verifier runs after Recommendation (not before, as a literal top-to-bottom
read of the M3 spec's agent list would suggest) to match the spec's own
target pipeline diagram, which places "Recommendation Engine -> Verifier ->
LLM Narrative Generation" in that order — verification only makes sense once
there are recommendations to verify.
"""

import time
from typing import Callable

from langgraph.graph import END, StateGraph

from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.orchestrator.state import AgentState


def _timed(stage_name: str, fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    def wrapped(state: AgentState) -> AgentState:
        start = time.perf_counter()
        new_state = fn(state)
        elapsed = round(time.perf_counter() - start, 4)
        latencies = dict(new_state.get("latencies", {}))
        latencies[stage_name] = elapsed
        return {**new_state, "latencies": latencies}

    return wrapped


def build_orchestrator(
    planner_agent: PlannerAgent,
    retrieval_agent: RetrievalAgent,
    knowledge_graph_agent: KnowledgeGraphAgent,
    evidence_fusion_agent: EvidenceFusionAgent,
    reasoning_agent: ReasoningAgent,
    recommendation_agent: RecommendationAgent,
    verifier_agent: VerifierAgent,
    report_agent: ReportAgent,
):
    graph = StateGraph(AgentState)
    graph.add_node("planner", _timed("planning", planner_agent.run))
    graph.add_node("retrieval", _timed("retrieval", retrieval_agent.run))
    graph.add_node("knowledge_graph", _timed("knowledge_graph", knowledge_graph_agent.run))
    graph.add_node("evidence_fusion", _timed("evidence_fusion", evidence_fusion_agent.run))
    graph.add_node("reasoning", _timed("reasoning", reasoning_agent.run))
    graph.add_node("recommendation", _timed("recommendation", recommendation_agent.run))
    graph.add_node("verifier", _timed("verification", verifier_agent.run))
    graph.add_node("report", _timed("narration", report_agent.run))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "knowledge_graph")
    graph.add_edge("knowledge_graph", "evidence_fusion")
    graph.add_edge("evidence_fusion", "reasoning")
    graph.add_edge("reasoning", "recommendation")
    graph.add_edge("recommendation", "verifier")
    graph.add_edge("verifier", "report")
    graph.add_edge("report", END)

    return graph.compile()
