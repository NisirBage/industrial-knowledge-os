"""Shared state passed between orchestrator nodes. No agent invokes another
directly — every agent reads what it needs from this state and writes its
output back into it (per M3's "communication occurs through shared state
objects" requirement).

Kept fields (query/evidence/answer/confidence/citations/recommended_actions)
are the backward-compatible surface ChatService/ChatResponse already read;
everything else is new M3 structure that ChatResponse also now exposes
(reasoning_summary, contradictions, recommendations, latencies).

O2 (Faithful Cognitive Reasoning) additions: `claim_graph` (the pruned,
SUPPORTED/PARTIALLY_SUPPORTED-only Claim Graph, propagated-confidence-
updated once Confidence Propagation runs), `full_verified_claim_graph`
(every extracted claim with its verification status set, nothing dropped —
what Faithfulness Scoring needs for true coverage/contradiction-rate
denominators), `recommendation_verification` (the full verified+rejected
record from Recommendation Verification), `faithfulness_report`.

O3 (Scientific Evaluation & Self-Improving Cognition) additions:
`reflection_report` — the Self-Reflection Engine's read-only analysis,
written by the very last pipeline stage; nothing downstream reads it.

O3.3 (Unified Execution Trace & Failure Intelligence Core) additions:
`execution_events` — the immutable Started/Finished `ExecutionEvent` pairs
every stage emits via the orchestrator's instrumentation wrapper, alongside
(not replacing) `latencies`. `ChatService` assembles the final
`ExecutionTrace` from this list once the graph completes.
"""

from typing import TypedDict

from app.claims.models import ClaimGraph
from app.contradiction.models import Contradiction
from app.entity_extraction.schemas import ExtractedEntity
from app.evidence.graph import EvidenceGraph
from app.evidence.models import EvidenceBundle
from app.execution.models import ExecutionEvent
from app.faithfulness.models import FaithfulnessReport
from app.planning.models import TaskGraph
from app.reasoning.graph_models import ReasoningGraph
from app.recommendation.models import RecommendationBundle
from app.recommendation.verification import VerifiedRecommendationBundle
from app.reflection.models import ReflectionReport
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

    # O2 additions.
    claim_graph: ClaimGraph
    full_verified_claim_graph: ClaimGraph
    recommendation_verification: VerifiedRecommendationBundle
    faithfulness_report: FaithfulnessReport

    # O3 additions.
    reflection_report: ReflectionReport

    # O3.3 additions.
    execution_events: list[ExecutionEvent]

    # Backward-compatible fields ChatService/ChatResponse read directly.
    evidence: list[Evidence]
    answer: str
    confidence: float
    citations: list[dict]
    recommended_actions: list[str]
