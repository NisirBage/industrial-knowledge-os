"""LangGraph wiring: planner -> retrieval -> knowledge_graph -> evidence_fusion
-> reasoning -> claim_extraction -> claim_verification -> confidence_propagation
-> recommendation -> verifier -> recommendation_verification -> report -> faithfulness.

Every node is wrapped with a latency timer (Observability requirement) that
records elapsed seconds into state["latencies"][stage_name], so per-stage
cost is measurable without touching each agent's own code.

O3.3 (Unified Execution Trace & Failure Intelligence Core) additions: the
same wrapper also emits a Started/Finished `ExecutionEvent` pair per stage
into state["execution_events"] via `ExecutionRecorder` — additive, alongside
the existing `latencies` dict, never replacing it. No agent's own code
knows this instrumentation exists ("no manual bookkeeping inside agents").

Verifier runs after Recommendation (not before, as a literal top-to-bottom
read of the M3 spec's agent list would suggest) to match the spec's own
target pipeline diagram, which places "Recommendation Engine -> Verifier ->
LLM Narrative Generation" in that order — verification only makes sense once
there are recommendations to verify.

O2 (Faithful Cognitive Reasoning) inserts a deterministic Cognitive
Verification Layer between Reasoning and the Narrator: Claim Extraction
decomposes evidence into atomic claims, Claim Verification checks each one
against the Knowledge Graph/corroborating evidence and drops unsupported
ones, Confidence Propagation spreads support across the (pruned) Claim
Graph, and Recommendation Verification applies the same evidence/confidence/
graph-support bar to recommendations after the existing structural
`verifier` check. Faithfulness runs last, scoring the Narrator's answer
against the verified claims that produced it — it needs the answer text, so
it cannot run any earlier.
"""

import time
from typing import Callable

from langgraph.graph import END, StateGraph

from app.agents.claim_extraction_agent import ClaimExtractionAgent
from app.agents.claim_verification_agent import ClaimVerificationAgent
from app.agents.confidence_propagation_agent import ConfidencePropagationAgent
from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.faithfulness_agent import FaithfulnessAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.recommendation_verification_agent import RecommendationVerificationAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.report_agent import ReportAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.execution.recorder import ExecutionRecorder, current_memory_bytes
from app.orchestrator.state import AgentState


def _timed(stage_name: str, fn: Callable[[AgentState], AgentState], component: str | None = None) -> Callable[[AgentState], AgentState]:
    component = component or stage_name

    def wrapped(state: AgentState) -> AgentState:
        query = state.get("query", "")
        memory_before = current_memory_bytes()
        started_event = ExecutionRecorder.record_started(component, query)

        start = time.perf_counter()
        new_state = fn(state)
        elapsed = round(time.perf_counter() - start, 4)

        latencies = dict(new_state.get("latencies", {}))
        latencies[stage_name] = elapsed

        produced_keys = sorted(set(new_state.keys()) - set(state.keys()))
        finished_event = ExecutionRecorder.record_finished(
            component, query, elapsed, produced_keys, new_state.get("confidence"), memory_before,
        )
        events = list(new_state.get("execution_events", [])) + [started_event, finished_event]

        return {**new_state, "latencies": latencies, "execution_events": events}

    return wrapped


def build_orchestrator(
    planner_agent: PlannerAgent,
    retrieval_agent: RetrievalAgent,
    knowledge_graph_agent: KnowledgeGraphAgent,
    evidence_fusion_agent: EvidenceFusionAgent,
    reasoning_agent: ReasoningAgent,
    claim_extraction_agent: ClaimExtractionAgent,
    claim_verification_agent: ClaimVerificationAgent,
    confidence_propagation_agent: ConfidencePropagationAgent,
    recommendation_agent: RecommendationAgent,
    verifier_agent: VerifierAgent,
    recommendation_verification_agent: RecommendationVerificationAgent,
    report_agent: ReportAgent,
    faithfulness_agent: FaithfulnessAgent,
    reflection_agent: ReflectionAgent,
):
    graph = StateGraph(AgentState)
    graph.add_node("planner", _timed("planning", planner_agent.run))
    graph.add_node("retrieval", _timed("retrieval", retrieval_agent.run))
    graph.add_node("knowledge_graph", _timed("knowledge_graph", knowledge_graph_agent.run))
    graph.add_node("evidence_fusion", _timed("evidence_fusion", evidence_fusion_agent.run))
    graph.add_node("reasoning", _timed("reasoning", reasoning_agent.run))
    graph.add_node("claim_extraction", _timed("claim_extraction", claim_extraction_agent.run))
    graph.add_node("claim_verification", _timed("claim_verification", claim_verification_agent.run))
    graph.add_node("confidence_propagation", _timed("confidence_propagation", confidence_propagation_agent.run))
    graph.add_node("recommendation", _timed("recommendation", recommendation_agent.run))
    graph.add_node("verifier", _timed("verification", verifier_agent.run))
    graph.add_node("verify_recommendations", _timed("recommendation_verification", recommendation_verification_agent.run))
    graph.add_node("report", _timed("narration", report_agent.run))
    graph.add_node("faithfulness", _timed("faithfulness", faithfulness_agent.run))
    graph.add_node("reflection", _timed("reflection", reflection_agent.run))

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "knowledge_graph")
    graph.add_edge("knowledge_graph", "evidence_fusion")
    graph.add_edge("evidence_fusion", "reasoning")
    graph.add_edge("reasoning", "claim_extraction")
    graph.add_edge("claim_extraction", "claim_verification")
    graph.add_edge("claim_verification", "confidence_propagation")
    graph.add_edge("confidence_propagation", "recommendation")
    graph.add_edge("recommendation", "verifier")
    graph.add_edge("verifier", "verify_recommendations")
    graph.add_edge("verify_recommendations", "report")
    graph.add_edge("report", "faithfulness")
    graph.add_edge("faithfulness", "reflection")
    graph.add_edge("reflection", END)

    return graph.compile()
