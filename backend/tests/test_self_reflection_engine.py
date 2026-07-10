from app.evidence.models import EvidenceBundle
from app.faithfulness.models import FaithfulnessReport
from app.recommendation.verification import RecommendationVerification, RecommendationVerificationStatus, VerifiedRecommendationBundle
from app.reflection.engine import SelfReflectionEngine
from app.retrieval.evidence import Evidence


def _bundle(evidence=None, missing_information=None):
    return EvidenceBundle(
        evidence=evidence or [], graph_evidence=[], timeline=[], metadata=[],
        missing_information=missing_information or [], confidence=0.7,
    )


def _faithfulness(overall=0.8):
    return FaithfulnessReport(
        overall_faithfulness=overall, sentence_faithfulness=[], recommendation_faithfulness=1.0,
        graph_faithfulness=1.0, claim_coverage=1.0, unsupported_claim_count=0,
        hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=0.0,
    )


def test_evidence_sufficient_when_enough_items_and_no_gaps():
    engine = SelfReflectionEngine()
    bundle = _bundle(evidence=[
        Evidence(document_id="d", page_number=1, chunk_id="c1", text="a"),
        Evidence(document_id="d", page_number=2, chunk_id="c2", text="b"),
    ])
    report = engine.reflect("query", bundle, 0.8, _faithfulness(), None)

    assert report.evidence_sufficient is True


def test_evidence_insufficient_when_too_few_items():
    engine = SelfReflectionEngine()
    bundle = _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="a")])
    report = engine.reflect("query", bundle, 0.8, _faithfulness(), None)

    assert report.evidence_sufficient is False


def test_evidence_insufficient_when_missing_information_present():
    engine = SelfReflectionEngine()
    bundle = _bundle(
        evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="a"), Evidence(document_id="d", page_number=2, chunk_id="c2", text="b")],
        missing_information=["no maintenance records found"],
    )
    report = engine.reflect("query", bundle, 0.8, _faithfulness(), None)

    assert report.evidence_sufficient is False


def test_confidence_not_justified_when_high_confidence_but_low_faithfulness():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(), 0.9, _faithfulness(overall=0.1), None)

    assert report.confidence_justified is False


def test_confidence_justified_when_high_confidence_and_high_faithfulness():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(), 0.9, _faithfulness(overall=0.9), None)

    assert report.confidence_justified is True


def test_confidence_justified_when_no_faithfulness_report_available():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(), 0.9, None, None)

    assert report.confidence_justified is True


def test_recommendations_verified_ratio_defaults_to_one_when_none_provided():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(), 0.5, _faithfulness(), None)

    assert report.recommendations_verified_ratio == 1.0


def test_recommendations_verified_ratio_reflects_real_verification_outcome():
    engine = SelfReflectionEngine()
    verification = VerifiedRecommendationBundle(
        verified=["r1"], rejected=["r2", "r3"],
        verifications={
            "r1": RecommendationVerification("r1", RecommendationVerificationStatus.VERIFIED),
            "r2": RecommendationVerification("r2", RecommendationVerificationStatus.REJECTED),
            "r3": RecommendationVerification("r3", RecommendationVerificationStatus.REJECTED),
        },
        overall_confidence=0.5,
    )
    report = engine.reflect("query", _bundle(), 0.5, _faithfulness(), verification)

    assert report.recommendations_verified_ratio == round(1 / 3, 4)


def test_additional_retrieval_suggested_when_evidence_sparse():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="a")]), 0.5, _faithfulness(), None)

    assert report.additional_retrieval_suggested is True


def test_additional_retrieval_not_suggested_when_evidence_plentiful():
    engine = SelfReflectionEngine()
    bundle = _bundle(evidence=[Evidence(document_id="d", page_number=i, chunk_id=f"c{i}", text="a") for i in range(5)])
    report = engine.reflect("query", bundle, 0.5, _faithfulness(), None)

    assert report.additional_retrieval_suggested is False


def test_summary_is_a_nonempty_string():
    engine = SelfReflectionEngine()
    report = engine.reflect("query", _bundle(), 0.5, _faithfulness(), None)

    assert isinstance(report.summary, str)
    assert len(report.summary) > 0


def test_reflection_never_mutates_inputs():
    engine = SelfReflectionEngine()
    bundle = _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="a")])
    original_missing = list(bundle.missing_information)

    engine.reflect("query", bundle, 0.5, _faithfulness(), None)

    assert bundle.missing_information == original_missing
