from app.execution.models import ExecutionEvent, ExecutionTrace
from app.execution.timeline import reconstruct_timeline


def _event(component, event_type, timestamp, duration=0.0, confidence=None, metadata=None):
    return ExecutionEvent(
        event_type=event_type, component=component, timestamp=timestamp, duration=duration,
        confidence=confidence, latency=duration, metadata=metadata or {},
    )


def _sequential_trace() -> ExecutionTrace:
    events = [
        _event("planning", "planning_started", "2026-01-01T00:00:00+00:00"),
        _event("planning", "planning_finished", "2026-01-01T00:00:01+00:00", duration=1.0, confidence=None),
        _event("retrieval", "retrieval_started", "2026-01-01T00:00:01+00:00"),
        _event("retrieval", "retrieval_finished", "2026-01-01T00:00:03+00:00", duration=2.0, confidence=0.7),
        _event("faithfulness", "faithfulness_started", "2026-01-01T00:00:03+00:00"),
        _event("faithfulness", "faithfulness_finished", "2026-01-01T00:00:04+00:00", duration=1.0, confidence=0.9),
    ]
    return ExecutionTrace(execution_id="e1", query="q", events=events)


def test_component_order_matches_chronological_start_order():
    timeline = reconstruct_timeline(_sequential_trace())
    assert timeline.component_order == ["planning", "retrieval", "faithfulness"]


def test_spans_carry_real_duration_and_confidence_from_finished_events():
    timeline = reconstruct_timeline(_sequential_trace())
    retrieval_span = next(s for s in timeline.spans if s.component == "retrieval")
    assert retrieval_span.duration_seconds == 2.0
    assert retrieval_span.confidence == 0.7


def test_sequential_pipeline_has_no_detected_parallel_branches():
    timeline = reconstruct_timeline(_sequential_trace())
    assert timeline.parallel_branches == []


def test_overlapping_spans_are_detected_as_parallel_branches():
    events = [
        _event("a", "a_started", "2026-01-01T00:00:00+00:00"),
        _event("a", "a_finished", "2026-01-01T00:00:05+00:00", duration=5.0),
        _event("b", "b_started", "2026-01-01T00:00:02+00:00"),
        _event("b", "b_finished", "2026-01-01T00:00:04+00:00", duration=2.0),
    ]
    trace = ExecutionTrace(execution_id="e1", query="q", events=events)
    timeline = reconstruct_timeline(trace)
    assert ("a", "b") in timeline.parallel_branches


def test_confidence_evolution_only_includes_events_with_real_confidence():
    timeline = reconstruct_timeline(_sequential_trace())
    assert timeline.confidence_evolution == [("retrieval", 0.7), ("faithfulness", 0.9)]


def test_dag_edges_form_a_chain_matching_execution_order():
    timeline = reconstruct_timeline(_sequential_trace())
    assert timeline.dag.nodes == ["planning", "retrieval", "faithfulness"]
    assert timeline.dag.edges == [("planning", "retrieval"), ("retrieval", "faithfulness")]


def test_cache_hits_extracted_from_event_metadata():
    events = [
        _event("caching", "caching_started", "2026-01-01T00:00:00+00:00"),
        _event("caching", "caching_finished", "2026-01-01T00:00:01+00:00", duration=0.01, metadata={"cache_hit": True}),
    ]
    trace = ExecutionTrace(execution_id="e1", query="q", events=events)
    timeline = reconstruct_timeline(trace)
    assert timeline.cache_hits == ["caching"]


def test_empty_trace_produces_empty_timeline():
    trace = ExecutionTrace(execution_id="e1", query="q")
    timeline = reconstruct_timeline(trace)
    assert timeline.spans == []
    assert timeline.component_order == []
    assert timeline.dag.nodes == []
    assert timeline.dag.edges == []
