"""O3.3 Component 10 migration adapter tests: `ErrorAnalyzer.analyze_trace()`
consumes an `ExecutionTrace` instead of the full domain objects `analyze()`
takes, using the exact same named thresholds (see analyzer.py)."""

from app.error_analysis.analyzer import ErrorAnalyzer
from app.error_analysis.models import FailureCategory, FailureSeverity
from app.execution.models import ExecutionTrace


def _trace(**subsystem_traces) -> ExecutionTrace:
    return ExecutionTrace(execution_id="e1", query="query", **subsystem_traces)


def test_no_failures_for_a_healthy_trace():
    trace = _trace(
        retrieval_trace={"evidence_count": 2, "confidence": 0.7, "missing_information_count": 0},
        claim_extraction_trace={"claim_count": 1},
        verification_trace={"claim_coverage": 1.0, "unsupported_claim_count": 0},
        nli_trace={"contradiction_rate": 0.0},
        faithfulness_trace={"overall_faithfulness": 0.9},
    )
    report = ErrorAnalyzer().analyze_trace(trace)
    assert not report.has_failures


def test_retrieval_failure_when_retrieval_trace_missing_or_empty():
    report = ErrorAnalyzer().analyze_trace(_trace())
    assert any(f.category == FailureCategory.RETRIEVAL_FAILURE for f in report.failures)


def test_ranking_failure_when_confidence_low_but_evidence_present():
    trace = _trace(retrieval_trace={"evidence_count": 3, "confidence": 0.05, "missing_information_count": 0})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert any(f.category == FailureCategory.RANKING_FAILURE for f in report.failures)
    assert not any(f.category == FailureCategory.RETRIEVAL_FAILURE for f in report.failures)


def test_claim_extraction_failure_when_evidence_present_but_no_claims():
    trace = _trace(retrieval_trace={"evidence_count": 2, "confidence": 0.8, "missing_information_count": 0})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert any(f.category == FailureCategory.CLAIM_EXTRACTION_FAILURE for f in report.failures)


def test_verification_failure_severity_matches_analyze_thresholds():
    trace = _trace(verification_trace={"claim_coverage": 0.0, "unsupported_claim_count": 9})
    report = ErrorAnalyzer().analyze_trace(trace)
    failure = next(f for f in report.failures if f.category == FailureCategory.VERIFICATION_FAILURE)
    assert failure.severity == FailureSeverity.CRITICAL


def test_nli_failure_when_contradiction_rate_high():
    trace = _trace(nli_trace={"contradiction_rate": 0.9})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert any(f.category == FailureCategory.NLI_FAILURE for f in report.failures)


def test_faithfulness_failure_when_score_low():
    trace = _trace(faithfulness_trace={"overall_faithfulness": 0.1})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert any(f.category == FailureCategory.FAITHFULNESS_FAILURE for f in report.failures)


def test_recommendation_failure_when_all_rejected():
    trace = _trace(recommendation_trace={"verified_count": 0, "rejected_count": 2})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert any(f.category == FailureCategory.RECOMMENDATION_FAILURE for f in report.failures)


def test_recommendation_failure_not_flagged_when_some_verified():
    trace = _trace(recommendation_trace={"verified_count": 1, "rejected_count": 1})
    report = ErrorAnalyzer().analyze_trace(trace)
    assert not any(f.category == FailureCategory.RECOMMENDATION_FAILURE for f in report.failures)


def test_planner_failure_severity_critical_when_all_nodes_failed():
    trace = _trace(planner_trace={"node_count": 2, "failed_node_count": 2})
    report = ErrorAnalyzer().analyze_trace(trace)
    failure = next(f for f in report.failures if f.category == FailureCategory.PLANNER_FAILURE)
    assert failure.severity == FailureSeverity.CRITICAL


def test_report_query_matches_trace_query():
    report = ErrorAnalyzer().analyze_trace(_trace(retrieval_trace={"evidence_count": 1, "confidence": 0.8, "missing_information_count": 0}, claim_extraction_trace={"claim_count": 1}))
    assert report.query == "query"
