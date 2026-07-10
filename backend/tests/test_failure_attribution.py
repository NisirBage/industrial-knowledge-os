from app.error_analysis.models import FailureSeverity
from app.execution.models import ExecutionTrace
from app.failure_intelligence.attribution import FailureAttributionEngine
from app.failure_intelligence.models import FailurePattern


def _pattern(affected_components=None, example_execution_ids=None) -> FailurePattern:
    return FailurePattern(
        pattern_id="p1", category="retrieval_failure", support=2, frequency=0.5,
        severity=FailureSeverity.HIGH,
        affected_components=["HybridRetriever"] if affected_components is None else affected_components,
        example_execution_ids=["e1", "e2"] if example_execution_ids is None else example_execution_ids,
        confidence=0.7, description="d",
    )


def test_primary_cause_is_first_affected_component():
    attribution = FailureAttributionEngine().attribute(_pattern(["HybridRetriever", "EvidenceFusionEngine"]))
    assert attribution.primary_cause == "HybridRetriever"
    assert attribution.secondary_cause == "EvidenceFusionEngine"


def test_contributing_components_are_everything_after_secondary():
    attribution = FailureAttributionEngine().attribute(_pattern(["A", "B", "C", "D"]))
    assert attribution.contributing_components == ["C", "D"]


def test_no_secondary_cause_when_only_one_component():
    attribution = FailureAttributionEngine().attribute(_pattern(["OnlyOne"]))
    assert attribution.secondary_cause is None
    assert attribution.contributing_components == []


def test_never_fabricates_cause_when_no_affected_components():
    attribution = FailureAttributionEngine().attribute(_pattern(affected_components=[]))
    assert attribution.primary_cause == "unattributed"
    assert attribution.evidence_confidence == 0.0


def test_severity_and_frequency_pass_through_from_pattern():
    attribution = FailureAttributionEngine().attribute(_pattern())
    assert attribution.severity == FailureSeverity.HIGH
    assert attribution.historical_frequency == 0.5


def test_execution_confidence_averages_matching_traces():
    traces = [
        ExecutionTrace(execution_id="e1", query="q", confidence=0.6),
        ExecutionTrace(execution_id="e2", query="q", confidence=0.4),
        ExecutionTrace(execution_id="e3", query="q", confidence=0.9),
    ]
    attribution = FailureAttributionEngine().attribute(_pattern(example_execution_ids=["e1", "e2"]), traces)
    assert attribution.execution_confidence == round((0.6 + 0.4) / 2, 4)


def test_execution_confidence_is_zero_when_no_traces_match():
    traces = [ExecutionTrace(execution_id="other", query="q", confidence=0.9)]
    attribution = FailureAttributionEngine().attribute(_pattern(example_execution_ids=["e1"]), traces)
    assert attribution.execution_confidence == 0.0


def test_execution_confidence_is_zero_when_no_traces_provided():
    attribution = FailureAttributionEngine().attribute(_pattern())
    assert attribution.execution_confidence == 0.0
