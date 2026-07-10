from app.error_analysis.models import DetectedFailure, FailureCategory, FailureSeverity
from app.execution.failure_event import FailureEvent


def _detected_failure() -> DetectedFailure:
    return DetectedFailure(
        category=FailureCategory.RETRIEVAL_FAILURE,
        root_cause="No evidence was retrieved for this query.",
        affected_components=["HybridRetriever"],
        confidence=0.9,
        suggested_improvement="Check ingestion coverage.",
        severity=FailureSeverity.CRITICAL,
    )


def test_from_detected_failure_references_the_execution_id():
    event = FailureEvent.from_detected_failure("exec-1", _detected_failure())

    assert event.execution_id == "exec-1"
    assert event.failure_type == FailureCategory.RETRIEVAL_FAILURE
    assert event.severity == FailureSeverity.CRITICAL
    assert event.confidence == 0.9
    assert event.root_cause == "No evidence was retrieved for this query."


def test_from_detected_failure_derives_component_from_affected_components():
    event = FailureEvent.from_detected_failure("exec-1", _detected_failure())
    assert event.component == "HybridRetriever"


def test_from_detected_failure_defaults_component_to_unknown_when_none_affected():
    failure = DetectedFailure(
        category=FailureCategory.PLANNER_FAILURE, root_cause="r", affected_components=[],
        confidence=0.5, suggested_improvement="s",
    )
    event = FailureEvent.from_detected_failure("exec-1", failure)
    assert event.component == "unknown"


def test_from_detected_failure_defaults_metadata_to_suggested_improvement():
    event = FailureEvent.from_detected_failure("exec-1", _detected_failure())
    assert event.metadata == {"suggested_improvement": "Check ingestion coverage."}


def test_from_detected_failure_accepts_explicit_evidence_claim_and_graph_ids():
    event = FailureEvent.from_detected_failure(
        "exec-1", _detected_failure(),
        evidence_ids=["e1", "e2"], claim_ids=["c1"], graph_nodes=["n1"],
    )
    assert event.evidence_ids == ["e1", "e2"]
    assert event.claim_ids == ["c1"]
    assert event.graph_nodes == ["n1"]


def test_each_failure_event_gets_a_unique_id():
    failure = _detected_failure()
    e1 = FailureEvent.from_detected_failure("exec-1", failure)
    e2 = FailureEvent.from_detected_failure("exec-1", failure)
    assert e1.id != e2.id


def test_failure_event_is_frozen():
    event = FailureEvent.from_detected_failure("exec-1", _detected_failure())
    try:
        event.component = "other"  # type: ignore[misc]
        assert False, "FailureEvent should be immutable"
    except AttributeError:
        pass
