from app.claims.models import Claim, ClaimGraph, ClaimType
from app.context_compression.models import CompressionStats
from app.error_analysis.analyzer import ErrorAnalyzer
from app.error_analysis.models import FailureCategory, FailureSeverity
from app.evidence.models import EvidenceBundle
from app.faithfulness.models import FaithfulnessReport
from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.recommendation.verification import RecommendationVerification, RecommendationVerificationStatus, VerifiedRecommendationBundle
from app.retrieval.evidence import Evidence


def _bundle(evidence=None, confidence=0.7):
    return EvidenceBundle(evidence=evidence or [], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=confidence)


def _faithfulness(overall=0.8, coverage=1.0, unsupported=0, contradiction_rate=0.0):
    return FaithfulnessReport(
        overall_faithfulness=overall, sentence_faithfulness=[], recommendation_faithfulness=1.0,
        graph_faithfulness=1.0, claim_coverage=coverage, unsupported_claim_count=unsupported,
        hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=contradiction_rate,
    )


def test_no_failures_detected_for_a_healthy_query():
    analyzer = ErrorAnalyzer()
    claim = Claim(id="c1", claim_type=ClaimType.FACT, text="Equipment = P-101", subject="P-101", predicate="is_a")
    report = analyzer.analyze(
        "query", None, _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running.")]),
        ClaimGraph(claims=[claim], relations=[]), _faithfulness(), None,
    )
    assert not report.has_failures


def test_retrieval_failure_detected_when_no_evidence():
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze("query", None, _bundle(evidence=[]), ClaimGraph(claims=[], relations=[]), None, None)

    assert report.has_failures
    assert report.failures[0].category == FailureCategory.RETRIEVAL_FAILURE


def test_ranking_failure_only_checked_when_evidence_exists():
    analyzer = ErrorAnalyzer()
    bundle = _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="text")], confidence=0.05)
    report = analyzer.analyze("query", None, bundle, ClaimGraph(claims=[], relations=[]), None, None)

    assert any(f.category == FailureCategory.RANKING_FAILURE for f in report.failures)
    assert not any(f.category == FailureCategory.RETRIEVAL_FAILURE for f in report.failures)


def test_planner_failure_detected_from_failed_task_node():
    analyzer = ErrorAnalyzer()
    task_graph = TaskGraph(nodes=[TaskNode(id="n1", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.FAILED)])
    report = analyzer.analyze("query", task_graph, _bundle(), ClaimGraph(claims=[], relations=[]), None, None)

    assert any(f.category == FailureCategory.PLANNER_FAILURE for f in report.failures)


def test_claim_extraction_failure_when_evidence_exists_but_no_claims():
    analyzer = ErrorAnalyzer()
    bundle = _bundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="text")])
    report = analyzer.analyze("query", None, bundle, ClaimGraph(claims=[], relations=[]), None, None)

    assert any(f.category == FailureCategory.CLAIM_EXTRACTION_FAILURE for f in report.failures)


def test_verification_failure_when_most_claims_dropped():
    analyzer = ErrorAnalyzer()
    faithfulness = _faithfulness(coverage=0.1, unsupported=9)
    report = analyzer.analyze("query", None, _bundle(), None, faithfulness, None)

    assert any(f.category == FailureCategory.VERIFICATION_FAILURE for f in report.failures)


def test_nli_failure_when_contradiction_rate_high():
    analyzer = ErrorAnalyzer()
    faithfulness = _faithfulness(contradiction_rate=0.9)
    report = analyzer.analyze("query", None, _bundle(), None, faithfulness, None)

    assert any(f.category == FailureCategory.NLI_FAILURE for f in report.failures)


def test_faithfulness_failure_when_score_low():
    analyzer = ErrorAnalyzer()
    faithfulness = _faithfulness(overall=0.1)
    report = analyzer.analyze("query", None, _bundle(), None, faithfulness, None)

    assert any(f.category == FailureCategory.FAITHFULNESS_FAILURE for f in report.failures)


def test_recommendation_failure_when_all_rejected():
    analyzer = ErrorAnalyzer()
    verification = VerifiedRecommendationBundle(
        verified=[], rejected=["r1", "r2"],
        verifications={"r1": RecommendationVerification("r1", RecommendationVerificationStatus.REJECTED), "r2": RecommendationVerification("r2", RecommendationVerificationStatus.REJECTED)},
        overall_confidence=0.0,
    )
    report = analyzer.analyze("query", None, _bundle(), None, None, verification)

    assert any(f.category == FailureCategory.RECOMMENDATION_FAILURE for f in report.failures)


def test_recommendation_failure_not_flagged_when_some_verified():
    analyzer = ErrorAnalyzer()
    verification = VerifiedRecommendationBundle(
        verified=["r1"], rejected=["r2"],
        verifications={"r1": RecommendationVerification("r1", RecommendationVerificationStatus.VERIFIED), "r2": RecommendationVerification("r2", RecommendationVerificationStatus.REJECTED)},
        overall_confidence=0.5,
    )
    report = analyzer.analyze("query", None, _bundle(), None, None, verification)

    assert not any(f.category == FailureCategory.RECOMMENDATION_FAILURE for f in report.failures)


def test_analyze_compression_flags_aggressive_ratio():
    analyzer = ErrorAnalyzer()
    stats = CompressionStats(
        original_count=20, compressed_count=1, exact_duplicate_count=0, near_duplicate_count=0, cluster_count=1,
        original_tokens_estimate=1000, compressed_tokens_estimate=50,
    )
    failure = analyzer.analyze_compression(stats)

    assert failure is not None
    assert failure.category == FailureCategory.COMPRESSION_FAILURE


def test_analyze_compression_returns_none_for_normal_ratio():
    analyzer = ErrorAnalyzer()
    stats = CompressionStats(
        original_count=10, compressed_count=8, exact_duplicate_count=1, near_duplicate_count=1, cluster_count=8,
        original_tokens_estimate=500, compressed_tokens_estimate=400,
    )
    assert analyzer.analyze_compression(stats) is None


def test_every_detected_failure_has_a_suggested_improvement():
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze("query", None, _bundle(evidence=[]), ClaimGraph(claims=[], relations=[]), _faithfulness(overall=0.1), None)
    assert report.has_failures
    for failure in report.failures:
        assert failure.suggested_improvement
        assert failure.affected_components


def test_retrieval_failure_is_critical_severity():
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze("query", None, _bundle(evidence=[]), ClaimGraph(claims=[], relations=[]), None, None)

    retrieval_failure = next(f for f in report.failures if f.category == FailureCategory.RETRIEVAL_FAILURE)
    assert retrieval_failure.severity == FailureSeverity.CRITICAL


def test_planner_failure_is_critical_when_every_node_failed():
    analyzer = ErrorAnalyzer()
    task_graph = TaskGraph(nodes=[TaskNode(id="n1", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.FAILED)])
    report = analyzer.analyze("query", task_graph, _bundle(), ClaimGraph(claims=[], relations=[]), None, None)

    planner_failure = next(f for f in report.failures if f.category == FailureCategory.PLANNER_FAILURE)
    assert planner_failure.severity == FailureSeverity.CRITICAL


def test_planner_failure_is_high_when_only_some_nodes_failed():
    analyzer = ErrorAnalyzer()
    task_graph = TaskGraph(nodes=[
        TaskNode(id="n1", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.FAILED),
        TaskNode(id="n2", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.COMPLETED),
    ])
    report = analyzer.analyze("query", task_graph, _bundle(), ClaimGraph(claims=[], relations=[]), None, None)

    planner_failure = next(f for f in report.failures if f.category == FailureCategory.PLANNER_FAILURE)
    assert planner_failure.severity == FailureSeverity.HIGH


def test_faithfulness_failure_is_critical_when_score_near_zero():
    analyzer = ErrorAnalyzer()
    faithfulness = _faithfulness(overall=0.05)
    report = analyzer.analyze("query", None, _bundle(), None, faithfulness, None)

    failure = next(f for f in report.failures if f.category == FailureCategory.FAITHFULNESS_FAILURE)
    assert failure.severity == FailureSeverity.CRITICAL


def test_faithfulness_failure_is_high_when_score_moderately_low():
    analyzer = ErrorAnalyzer()
    faithfulness = _faithfulness(overall=0.3)
    report = analyzer.analyze("query", None, _bundle(), None, faithfulness, None)

    failure = next(f for f in report.failures if f.category == FailureCategory.FAITHFULNESS_FAILURE)
    assert failure.severity == FailureSeverity.HIGH


def test_detected_failure_defaults_to_frequency_one():
    analyzer = ErrorAnalyzer()
    report = analyzer.analyze("query", None, _bundle(evidence=[]), ClaimGraph(claims=[], relations=[]), None, None)

    assert all(f.frequency == 1 for f in report.failures)
