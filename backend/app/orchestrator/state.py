"""Shared state passed between orchestrator nodes. No agent invokes another
directly — every agent reads what it needs from this state and writes its
output back into it (per M3's "communication occurs through shared state
objects" requirement).

Kept fields (query/evidence/answer/confidence/citations/recommended_actions)
are the backward-compatible surface ChatService/ChatResponse already read;
everything else is new M3 structure that ChatResponse also now exposes
(reasoning_summary, contradictions, recommendations, latencies).
"""

from typing import TypedDict

from app.contradiction.models import Contradiction
from app.entity_extraction.schemas import ExtractedEntity
from app.evidence.graph import EvidenceGraph
from app.evidence.models import EvidenceBundle
from app.planning.models import TaskGraph
from app.reasoning.graph_models import ReasoningGraph
from app.recommendation.models import RecommendationBundle
from app.retrieval.evidence import Evidence


class AgentState(TypedDict, total=False):
    query: str
    query_entities: list[ExtractedEntity]
    plan: str
    task_graph: TaskGraph
    graph_evidence: list[Evidence]
    evidence_bundle: EvidenceBundle
    evidence_graph: EvidenceGraph
    reasoning_graph: ReasoningGraph
    contradictions: list[Contradiction]
    recommendation_bundle: RecommendationBundle
    verification_notes: list[str]
    latencies: dict[str, float]

    # Backward-compatible fields ChatService/ChatResponse read directly.
    evidence: list[Evidence]
    answer: str
    confidence: float
    citations: list[dict]
    recommended_actions: list[str]
