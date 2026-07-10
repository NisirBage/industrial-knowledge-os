"""Chat service: runs a query through the full agent orchestration graph
(planner -> retrieval -> knowledge_graph -> evidence_fusion -> reasoning ->
claim_extraction -> claim_verification -> confidence_propagation ->
recommendation -> verifier -> recommendation_verification -> report ->
faithfulness -> reflection) and shapes the result into the API response
schema.

O3.3 (Unified Execution Trace & Failure Intelligence Core): if constructed
with an `execution_repository`, every answer's `execution_events` +
`latencies` + per-subsystem signals are assembled into an `ExecutionTrace`
and persisted — additive, optional, and never allowed to break the chat
response (persistence failures are swallowed; this is an observability
side-channel, not part of the answer contract)."""

import asyncio
import dataclasses
import time

from app.claims.attribution import EvidenceAttributionBuilder
from app.claims.models import ClaimGraph
from app.execution.recorder import ExecutionRecorder
from app.execution.repository import ExecutionRepositoryProtocol
from app.execution.state_adapter import build_subsystem_traces
from app.retrieval.explain import explain_evidence
from app.schemas.chat import (
    ChatResponse,
    CitationOut,
    ClaimOut,
    ContradictionOut,
    EvidenceOut,
    FaithfulnessOut,
    RecommendationOut,
    ReflectionReportOut,
    SentenceAttributionOut,
    SentenceFaithfulnessOut,
)


class ChatService:
    def __init__(self, orchestrator, graph_repository, execution_repository: ExecutionRepositoryProtocol | None = None):
        self._orchestrator = orchestrator
        self._attribution_builder = EvidenceAttributionBuilder(graph_repository)
        self._execution_repository = execution_repository

    def answer(self, query: str) -> ChatResponse:
        start = time.perf_counter()
        started_at = ExecutionRecorder.now()
        final_state = self._orchestrator.invoke({"query": query})
        completed_at = ExecutionRecorder.now()
        overall_latency = round(time.perf_counter() - start, 4)

        latencies = dict(final_state.get("latencies", {}))
        latencies["overall"] = overall_latency

        reasoning_graph = final_state.get("reasoning_graph")
        recommendation_bundle = final_state.get("recommendation_bundle")
        claim_graph = final_state.get("claim_graph")
        faithfulness_report = final_state.get("faithfulness_report")
        recommendation_verification = final_state.get("recommendation_verification")
        reflection_report = final_state.get("reflection_report")

        self._persist_execution_trace(query, final_state, started_at, completed_at)

        return ChatResponse(
            summary=final_state.get("answer", ""),
            confidence=final_state.get("confidence", 0.0),
            citations=[CitationOut(**c) for c in final_state.get("citations", [])],
            evidence=[
                EvidenceOut(**dataclasses.asdict(e), final_rank=rank, explanation=explain_evidence(e, rank))
                for rank, e in enumerate(final_state.get("evidence", []), start=1)
            ],
            recommended_actions=final_state.get("recommended_actions", []),
            reasoning_summary=reasoning_graph.summary if reasoning_graph else "",
            contradictions=[ContradictionOut(**dataclasses.asdict(c)) for c in final_state.get("contradictions", [])],
            recommendations=[
                RecommendationOut(**dataclasses.asdict(r), **self._recommendation_verification_fields(r.id, recommendation_verification))
                for r in (recommendation_bundle.recommendations if recommendation_bundle else [])
            ],
            latencies=latencies,
            claims=[
                ClaimOut(
                    id=c.id,
                    claim_type=c.claim_type.value,
                    text=c.text,
                    subject=c.subject,
                    predicate=c.predicate,
                    object=c.object,
                    confidence=c.confidence,
                    propagated_confidence=c.propagated_confidence,
                    source_evidence_ids=c.source_evidence_ids,
                    verification_status=c.verification_status.value,
                    verification_notes=c.verification_notes,
                    support_count=c.support_count,
                    contradiction_count=c.contradiction_count,
                )
                for c in (claim_graph.claims if claim_graph else [])
            ],
            faithfulness=self._faithfulness_out(faithfulness_report),
            sentence_attributions=self._sentence_attributions(faithfulness_report, claim_graph),
            reflection=self._reflection_out(reflection_report),
        )

    def _persist_execution_trace(self, query: str, final_state: dict, started_at: str, completed_at: str) -> None:
        if self._execution_repository is None:
            return
        try:
            trace = ExecutionRecorder.finalize(
                execution_id=ExecutionRecorder.new_execution_id(),
                query=query,
                events=final_state.get("execution_events", []),
                latencies=final_state.get("latencies", {}),
                confidence=final_state.get("confidence", 0.0),
                started_at=started_at,
                completed_at=completed_at,
                subsystem_traces=build_subsystem_traces(final_state),
            )
            asyncio.run(self._execution_repository.save_trace(trace))
        except Exception:
            # Observability-only side-channel: a persistence failure must
            # never break the chat answer itself.
            pass

    def _sentence_attributions(self, faithfulness_report, claim_graph: ClaimGraph | None) -> list[SentenceAttributionOut]:
        if faithfulness_report is None or claim_graph is None:
            return []
        attributions = self._attribution_builder.build(faithfulness_report.sentence_faithfulness, claim_graph)
        return [SentenceAttributionOut(**dataclasses.asdict(a)) for a in attributions]

    @staticmethod
    def _recommendation_verification_fields(recommendation_id: str, recommendation_verification) -> dict:
        if recommendation_verification is None:
            return {}
        record = recommendation_verification.verifications.get(recommendation_id)
        if record is None:
            return {}
        return {"verification_status": record.status.value, "verification_reasons": record.reasons}

    @staticmethod
    def _faithfulness_out(faithfulness_report) -> FaithfulnessOut | None:
        if faithfulness_report is None:
            return None
        return FaithfulnessOut(
            overall_faithfulness=faithfulness_report.overall_faithfulness,
            sentence_faithfulness=[
                SentenceFaithfulnessOut(**dataclasses.asdict(s)) for s in faithfulness_report.sentence_faithfulness
            ],
            recommendation_faithfulness=faithfulness_report.recommendation_faithfulness,
            graph_faithfulness=faithfulness_report.graph_faithfulness,
            claim_coverage=faithfulness_report.claim_coverage,
            unsupported_claim_count=faithfulness_report.unsupported_claim_count,
            hallucinated_claim_count=faithfulness_report.hallucinated_claim_count,
            missing_evidence_count=faithfulness_report.missing_evidence_count,
            contradiction_rate=faithfulness_report.contradiction_rate,
            summary=faithfulness_report.summary,
        )

    @staticmethod
    def _reflection_out(reflection_report) -> ReflectionReportOut | None:
        if reflection_report is None:
            return None
        return ReflectionReportOut(**dataclasses.asdict(reflection_report))
