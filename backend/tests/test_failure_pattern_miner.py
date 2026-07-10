from app.execution.models import ExecutionEvent, ExecutionTrace
from app.failure_intelligence.miner import FailurePatternMiner


def _sequential_events(components: list[str]) -> list[ExecutionEvent]:
    events = []
    for i, component in enumerate(components):
        events.append(ExecutionEvent(event_type=f"{component}_started", component=component, timestamp=f"2026-01-01T00:00:{i:02d}+00:00", duration=0.0))
        events.append(ExecutionEvent(event_type=f"{component}_finished", component=component, timestamp=f"2026-01-01T00:00:{i+1:02d}+00:00", duration=1.0))
    return events


def _retrieval_failure_trace(execution_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        execution_id=execution_id, query="q", events=_sequential_events(["planning", "retrieval"]),
        confidence=0.1, retrieval_trace=None,
    )


def _healthy_trace(execution_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        execution_id=execution_id, query="q", events=_sequential_events(["planning", "retrieval"]),
        confidence=0.8, retrieval_trace={"evidence_count": 3, "confidence": 0.8, "missing_information_count": 0},
        claim_extraction_trace={"claim_count": 2}, verification_trace={"claim_coverage": 1.0, "unsupported_claim_count": 0},
        nli_trace={"contradiction_rate": 0.0}, faithfulness_trace={"overall_faithfulness": 0.9},
    )


def test_mine_returns_empty_list_for_no_traces():
    assert FailurePatternMiner().mine([]) == []


def test_no_patterns_when_failures_below_min_support():
    patterns = FailurePatternMiner(min_support=2).mine([_retrieval_failure_trace("e1"), _healthy_trace("e2"), _healthy_trace("e3")])
    assert not any(p.category == "retrieval_failure" for p in patterns)


def test_retrieval_failure_pattern_detected_when_support_met():
    patterns = FailurePatternMiner(min_support=2).mine([_retrieval_failure_trace("e1"), _retrieval_failure_trace("e2"), _healthy_trace("e3")])

    pattern = next(p for p in patterns if p.category == "retrieval_failure")
    assert pattern.support == 2
    assert pattern.frequency == round(2 / 3, 4)
    assert set(pattern.example_execution_ids) == {"e1", "e2"}
    assert pattern.affected_components == ["HybridRetriever"]


def test_pattern_ids_are_unique_across_mining_runs():
    traces = [_retrieval_failure_trace("e1"), _retrieval_failure_trace("e2")]
    first_run = FailurePatternMiner(min_support=2).mine(traces)
    second_run = FailurePatternMiner(min_support=2).mine(traces)
    assert first_run[0].pattern_id != second_run[0].pattern_id


def test_confidence_collapse_pattern_detected_across_traces():
    def collapsing_trace(execution_id: str) -> ExecutionTrace:
        return ExecutionTrace(
            execution_id=execution_id, query="q", events=_sequential_events(["planning", "retrieval"]),
            confidence=0.1, retrieval_trace={"evidence_count": 3, "confidence": 0.9, "missing_information_count": 0},
        )

    patterns = FailurePatternMiner(min_support=2).mine([collapsing_trace("e1"), collapsing_trace("e2"), _healthy_trace("e3")])
    pattern = next(p for p in patterns if p.category == "confidence_collapse")
    assert pattern.support == 2
    assert "ConfidencePropagation" in pattern.affected_components


def test_no_confidence_collapse_pattern_when_drop_is_small():
    def small_drop_trace(execution_id: str) -> ExecutionTrace:
        return ExecutionTrace(
            execution_id=execution_id, query="q", events=_sequential_events(["planning", "retrieval"]),
            confidence=0.75, retrieval_trace={"evidence_count": 3, "confidence": 0.8, "missing_information_count": 0},
        )

    patterns = FailurePatternMiner(min_support=2).mine([small_drop_trace("e1"), small_drop_trace("e2")])
    assert not any(p.category == "confidence_collapse" for p in patterns)


def test_repeated_failure_sequence_pattern_detected():
    patterns = FailurePatternMiner(min_support=2).mine([_retrieval_failure_trace("e1"), _retrieval_failure_trace("e2")])
    sequence_pattern = next(p for p in patterns if p.category == "repeated_failure_sequence")
    assert sequence_pattern.affected_components == ["planning", "retrieval"]
    assert sequence_pattern.support == 2


def test_every_pattern_has_a_nonempty_description():
    patterns = FailurePatternMiner(min_support=2).mine([_retrieval_failure_trace("e1"), _retrieval_failure_trace("e2")])
    assert patterns
    for pattern in patterns:
        assert pattern.description
