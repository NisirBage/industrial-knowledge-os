"""O3.3 Component 10 migration adapter tests: `SelfReflectionEngine.reflect_trace()`
consumes an `ExecutionTrace` instead of the full domain objects `reflect()`
takes, using the exact same named thresholds (see reflection/engine.py)."""

from app.execution.models import ExecutionTrace
from app.reflection.engine import SelfReflectionEngine


def _trace(confidence=0.5, **subsystem_traces) -> ExecutionTrace:
    return ExecutionTrace(execution_id="e1", query="query", confidence=confidence, **subsystem_traces)


def test_evidence_sufficient_when_enough_items_and_no_gaps():
    trace = _trace(retrieval_trace={"evidence_count": 2, "confidence": 0.7, "missing_information_count": 0})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.evidence_sufficient is True


def test_evidence_insufficient_when_too_few_items():
    trace = _trace(retrieval_trace={"evidence_count": 1, "confidence": 0.7, "missing_information_count": 0})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.evidence_sufficient is False


def test_evidence_insufficient_when_missing_information_present():
    trace = _trace(retrieval_trace={"evidence_count": 2, "confidence": 0.7, "missing_information_count": 1})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.evidence_sufficient is False


def test_confidence_not_justified_when_high_confidence_but_low_faithfulness():
    trace = _trace(confidence=0.9, faithfulness_trace={"overall_faithfulness": 0.1})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.confidence_justified is False


def test_confidence_justified_when_high_confidence_and_high_faithfulness():
    trace = _trace(confidence=0.9, faithfulness_trace={"overall_faithfulness": 0.9})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.confidence_justified is True


def test_confidence_justified_when_no_faithfulness_trace_available():
    trace = _trace(confidence=0.9)
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.confidence_justified is True


def test_recommendations_verified_ratio_defaults_to_one_when_no_recommendation_trace():
    report = SelfReflectionEngine().reflect_trace(_trace())
    assert report.recommendations_verified_ratio == 1.0


def test_recommendations_verified_ratio_reflects_trace_counts():
    trace = _trace(recommendation_trace={"verified_count": 1, "rejected_count": 2})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.recommendations_verified_ratio == round(1 / 3, 4)


def test_additional_retrieval_suggested_when_evidence_sparse():
    trace = _trace(retrieval_trace={"evidence_count": 1, "confidence": 0.5, "missing_information_count": 0})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.additional_retrieval_suggested is True


def test_additional_retrieval_not_suggested_when_evidence_plentiful():
    trace = _trace(retrieval_trace={"evidence_count": 5, "confidence": 0.5, "missing_information_count": 0})
    report = SelfReflectionEngine().reflect_trace(trace)
    assert report.additional_retrieval_suggested is False
