from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.core.enums import Priority, Severity
from app.recommendation.models import Recommendation, RecommendationBundle
from app.recommendation.verification import (
    RecommendationVerificationStatus,
    RecommendationVerificationThresholds,
    RecommendationVerifier,
)


def _recommendation(rec_id: str = "risk:c1", confidence: float = 0.7, supporting_evidence: list[str] | None = None) -> Recommendation:
    return Recommendation(
        id=rec_id,
        description="Investigate and address: recurring risk",
        priority=Priority.HIGH,
        severity=Severity.HIGH,
        estimated_impact="Potential unplanned downtime",
        required_personnel=["Maintenance Technician"],
        required_equipment=[],
        required_documents=[],
        estimated_downtime="TBD",
        confidence=confidence,
        justification="Derived from the reasoning graph.",
        supporting_evidence=supporting_evidence if supporting_evidence is not None else ["c1"],
    )


def _claim(claim_id: str, chunk_id: str = "c1", status: ClaimVerificationStatus = ClaimVerificationStatus.SUPPORTED, claim_type: ClaimType = ClaimType.FACT) -> Claim:
    return Claim(
        id=claim_id,
        claim_type=claim_type,
        text=claim_id,
        subject="P-101",
        predicate="is_a",
        source_evidence_ids=[chunk_id],
        verification_status=status,
    )


def test_recommendation_verified_when_all_thresholds_pass():
    recommendation = _recommendation()
    claim_graph = ClaimGraph(claims=[_claim("fact1", status=ClaimVerificationStatus.SUPPORTED)], relations=[])
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.7), claim_graph)

    assert len(result.verified) == 1
    assert result.rejected == []
    record = result.verifications["risk:c1"]
    assert record.status == RecommendationVerificationStatus.VERIFIED
    assert "fact1" in record.supporting_claim_ids


def test_recommendation_rejected_when_confidence_below_threshold():
    recommendation = _recommendation(confidence=0.1)
    claim_graph = ClaimGraph(claims=[_claim("fact1")], relations=[])
    verifier = RecommendationVerifier(RecommendationVerificationThresholds(min_confidence=0.3))

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.1), claim_graph)

    assert result.verified == []
    assert len(result.rejected) == 1
    record = result.verifications["risk:c1"]
    assert record.status == RecommendationVerificationStatus.REJECTED
    assert any("Confidence" in reason for reason in record.reasons)


def test_recommendation_rejected_when_no_evidence():
    recommendation = _recommendation(supporting_evidence=[])
    claim_graph = ClaimGraph(claims=[], relations=[])
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.7), claim_graph)

    assert result.rejected == [recommendation]
    assert any("evidence item" in reason for reason in result.verifications["risk:c1"].reasons)


def test_recommendation_rejected_when_no_claim_reaches_supported_status():
    recommendation = _recommendation()
    claim_graph = ClaimGraph(claims=[_claim("fact1", status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE)], relations=[])
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.7), claim_graph)

    assert result.rejected == [recommendation]
    record = result.verifications["risk:c1"]
    assert any("verification status" in reason for reason in record.reasons)


def test_graph_support_requirement_can_be_disabled_via_thresholds():
    recommendation = _recommendation()
    claim_graph = ClaimGraph(claims=[], relations=[])
    verifier = RecommendationVerifier(RecommendationVerificationThresholds(require_graph_support=False))

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.7), claim_graph)

    assert len(result.verified) == 1


def test_supporting_risk_claim_ids_only_includes_risk_and_hypothesis_types():
    recommendation = _recommendation()
    claim_graph = ClaimGraph(
        claims=[
            _claim("fact1", claim_type=ClaimType.FACT),
            _claim("risk1", claim_type=ClaimType.RISK),
            _claim("hyp1", claim_type=ClaimType.HYPOTHESIS),
        ],
        relations=[],
    )
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.7), claim_graph)
    record = result.verifications["risk:c1"]

    assert set(record.supporting_risk_claim_ids) == {"risk1", "hyp1"}
    assert set(record.supporting_claim_ids) == {"fact1", "risk1", "hyp1"}


def test_overall_confidence_averages_only_verified_recommendations():
    good = _recommendation(rec_id="good", confidence=0.8, supporting_evidence=["c1"])
    bad = _recommendation(rec_id="bad", confidence=0.1, supporting_evidence=["c1"])
    claim_graph = ClaimGraph(claims=[_claim("fact1")], relations=[])
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[good, bad], overall_confidence=0.45), claim_graph)

    assert len(result.verified) == 1
    assert result.overall_confidence == 0.8


def test_confidence_exactly_at_threshold_passes():
    recommendation = _recommendation(confidence=0.3)
    claim_graph = ClaimGraph(claims=[_claim("fact1")], relations=[])
    verifier = RecommendationVerifier(RecommendationVerificationThresholds(min_confidence=0.3))

    result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=0.3), claim_graph)

    assert len(result.verified) == 1


def test_empty_recommendation_bundle_produces_empty_verified_and_rejected():
    verifier = RecommendationVerifier()
    result = verifier.verify(RecommendationBundle(recommendations=[], overall_confidence=0.5), ClaimGraph(claims=[], relations=[]))

    assert result.verified == []
    assert result.rejected == []
    assert result.verifications == {}


def test_overall_confidence_is_zero_when_nothing_verified():
    bad = _recommendation(confidence=0.05)
    verifier = RecommendationVerifier()

    result = verifier.verify(RecommendationBundle(recommendations=[bad], overall_confidence=0.05), ClaimGraph(claims=[], relations=[]))

    assert result.verified == []
    assert result.overall_confidence == 0.0
