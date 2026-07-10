from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.contradiction.models import Contradiction
from app.core.enums import Priority, Severity
from app.faithfulness.models import FaithfulnessReport, SentenceFaithfulness
from app.recommendation.models import Recommendation, RecommendationBundle
from app.recommendation.verification import (
    RecommendationVerification,
    RecommendationVerificationStatus,
    VerifiedRecommendationBundle,
)
from app.repositories.graph_repository import InMemoryGraphRepository
from app.retrieval.evidence import Evidence
from app.services.chat_service import ChatService


class _FakeOrchestrator:
    def __init__(self, final_state: dict):
        self._final_state = final_state

    def invoke(self, initial_state: dict) -> dict:
        return {**initial_state, **self._final_state}


def _recommendation(rec_id: str = "r1") -> Recommendation:
    return Recommendation(
        id=rec_id,
        description="Investigate bearing wear",
        priority=Priority.HIGH,
        severity=Severity.HIGH,
        estimated_impact="downtime risk",
        required_personnel=["Technician"],
        required_equipment=[],
        required_documents=[],
        estimated_downtime="TBD",
        confidence=0.8,
        justification="derived from risk node",
        supporting_evidence=["c1"],
    )


def _claim(claim_id: str = "c1") -> Claim:
    return Claim(
        id=claim_id,
        claim_type=ClaimType.FACT,
        text="Equipment = P-101",
        subject="P-101",
        predicate="is_a",
        source_evidence_ids=["c1"],
        verification_status=ClaimVerificationStatus.SUPPORTED,
    )


def test_answer_with_minimal_state_produces_empty_o2_fields():
    service = ChatService(_FakeOrchestrator({"answer": "ok", "confidence": 0.5}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.claims == []
    assert response.faithfulness is None
    assert response.sentence_attributions == []


def test_answer_maps_claim_graph_into_claims_out():
    claim_graph = ClaimGraph(claims=[_claim()], relations=[])
    service = ChatService(_FakeOrchestrator({"claim_graph": claim_graph}), InMemoryGraphRepository())

    response = service.answer("query")

    assert len(response.claims) == 1
    assert response.claims[0].id == "c1"
    assert response.claims[0].claim_type == "fact"
    assert response.claims[0].verification_status == "supported"


def test_answer_maps_faithfulness_report_into_faithfulness_out():
    report = FaithfulnessReport(
        overall_faithfulness=0.8,
        sentence_faithfulness=[SentenceFaithfulness(sentence="a", entailed=True, best_claim_id="c1", score=0.9)],
        recommendation_faithfulness=1.0,
        graph_faithfulness=1.0,
        claim_coverage=1.0,
        unsupported_claim_count=0,
        hallucinated_claim_count=0,
        missing_evidence_count=0,
        contradiction_rate=0.0,
        summary="Faithfulness 0.80",
    )
    service = ChatService(_FakeOrchestrator({"faithfulness_report": report}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.faithfulness is not None
    assert response.faithfulness.overall_faithfulness == 0.8
    assert response.faithfulness.summary == "Faithfulness 0.80"
    assert len(response.faithfulness.sentence_faithfulness) == 1


def test_sentence_attributions_require_both_claim_graph_and_faithfulness_report():
    claim_graph = ClaimGraph(claims=[_claim()], relations=[])
    only_claims = ChatService(_FakeOrchestrator({"claim_graph": claim_graph}), InMemoryGraphRepository())
    assert only_claims.answer("q").sentence_attributions == []

    report = FaithfulnessReport(
        overall_faithfulness=0.5, sentence_faithfulness=[SentenceFaithfulness("a", True, "c1", 0.9)],
        recommendation_faithfulness=1.0, graph_faithfulness=1.0, claim_coverage=1.0,
        unsupported_claim_count=0, hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=0.0,
    )
    only_faithfulness = ChatService(_FakeOrchestrator({"faithfulness_report": report}), InMemoryGraphRepository())
    assert only_faithfulness.answer("q").sentence_attributions == []


def test_sentence_attributions_populated_when_both_present():
    claim_graph = ClaimGraph(claims=[_claim()], relations=[])
    report = FaithfulnessReport(
        overall_faithfulness=0.9, sentence_faithfulness=[SentenceFaithfulness("Equipment = P-101", True, "c1", 0.9)],
        recommendation_faithfulness=1.0, graph_faithfulness=1.0, claim_coverage=1.0,
        unsupported_claim_count=0, hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=0.0,
    )
    service = ChatService(
        _FakeOrchestrator({"claim_graph": claim_graph, "faithfulness_report": report}), InMemoryGraphRepository()
    )

    response = service.answer("query")

    assert len(response.sentence_attributions) == 1
    assert response.sentence_attributions[0].supporting_claim_ids == ["c1"]


def test_recommendation_gets_verified_status_when_record_exists():
    bundle = RecommendationBundle(recommendations=[_recommendation("r1")], overall_confidence=0.8)
    verification = VerifiedRecommendationBundle(
        verified=[_recommendation("r1")],
        rejected=[],
        verifications={"r1": RecommendationVerification("r1", RecommendationVerificationStatus.VERIFIED, ["ok"])},
        overall_confidence=0.8,
    )
    service = ChatService(
        _FakeOrchestrator({"recommendation_bundle": bundle, "recommendation_verification": verification}),
        InMemoryGraphRepository(),
    )

    response = service.answer("query")

    assert response.recommendations[0].verification_status == "verified"
    assert response.recommendations[0].verification_reasons == ["ok"]


def test_recommendation_defaults_to_unknown_when_no_verification_record():
    bundle = RecommendationBundle(recommendations=[_recommendation("r1")], overall_confidence=0.8)
    service = ChatService(_FakeOrchestrator({"recommendation_bundle": bundle}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.recommendations[0].verification_status == "unknown"
    assert response.recommendations[0].verification_reasons == []


def test_recommendation_defaults_to_unknown_when_id_missing_from_verifications_map():
    bundle = RecommendationBundle(recommendations=[_recommendation("r-not-in-map")], overall_confidence=0.8)
    verification = VerifiedRecommendationBundle(
        verified=[], rejected=[], verifications={"other-id": RecommendationVerification("other-id", RecommendationVerificationStatus.REJECTED)}, overall_confidence=0.0
    )
    service = ChatService(
        _FakeOrchestrator({"recommendation_bundle": bundle, "recommendation_verification": verification}),
        InMemoryGraphRepository(),
    )

    response = service.answer("query")

    assert response.recommendations[0].verification_status == "unknown"


def test_answer_includes_overall_latency_and_maps_evidence_contradictions():
    evidence = [Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running.")]
    contradiction = Contradiction(
        conflicting_evidence=["c1", "c2"], confidence=0.9, severity=Severity.HIGH,
        resolution_strategy="verify", recommended_clarification="check", description="conflict",
    )
    service = ChatService(
        _FakeOrchestrator({"evidence": evidence, "contradictions": [contradiction], "citations": [
            {"index": 1, "chunk_id": "c1", "document_id": "d1", "page_number": 1, "source": "fusion", "entity_ids": []}
        ]}),
        InMemoryGraphRepository(),
    )

    response = service.answer("query")

    assert "overall" in response.latencies
    assert len(response.evidence) == 1
    assert response.evidence[0].final_rank == 1
    assert response.evidence[0].explanation.startswith("Rank #1.")
    assert len(response.contradictions) == 1
    assert response.contradictions[0].description == "conflict"
    assert len(response.citations) == 1


def test_answer_maps_multiple_claims_preserving_order():
    claim_graph = ClaimGraph(claims=[_claim("c1"), _claim("c2")], relations=[])
    service = ChatService(_FakeOrchestrator({"claim_graph": claim_graph}), InMemoryGraphRepository())

    response = service.answer("query")

    assert [c.id for c in response.claims] == ["c1", "c2"]


def test_answer_with_no_recommendation_bundle_produces_empty_recommendations():
    service = ChatService(_FakeOrchestrator({}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.recommendations == []


def test_answer_recommended_actions_defaults_to_empty_list():
    service = ChatService(_FakeOrchestrator({}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.recommended_actions == []


def test_answer_defaults_summary_and_confidence_when_absent():
    service = ChatService(_FakeOrchestrator({}), InMemoryGraphRepository())

    response = service.answer("query")

    assert response.summary == ""
    assert response.confidence == 0.0


def test_answer_persists_execution_trace_when_repository_provided(tmp_path):
    import asyncio

    from app.execution.repository import SqlExecutionRepository

    db_path = tmp_path / "execution_chat_test.db"
    repository = SqlExecutionRepository(f"sqlite+aiosqlite:///{db_path}")
    asyncio.run(repository.init_models())

    # ChatService.answer() is always called synchronously (the real FastAPI
    # route is a sync `def`, run in a threadpool with no event loop already
    # active) — this test matches that exact calling convention rather than
    # calling `.answer()` from inside an already-running async test.
    service = ChatService(
        _FakeOrchestrator({"answer": "ok", "confidence": 0.6, "execution_events": []}),
        InMemoryGraphRepository(),
        execution_repository=repository,
    )

    response = service.answer("why did P-101 fail?")
    assert response.summary == "ok"

    traces = asyncio.run(repository.list_traces())
    assert len(traces) == 1
    assert traces[0].query == "why did P-101 fail?"
    assert traces[0].confidence == 0.6


def test_answer_does_not_raise_when_no_execution_repository_provided():
    service = ChatService(_FakeOrchestrator({"answer": "ok"}), InMemoryGraphRepository())
    response = service.answer("query")
    assert response.summary == "ok"
