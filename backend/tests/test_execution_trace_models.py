from app.execution.models import ExecutionEvent, ExecutionTrace


def _event(component: str, event_type: str, confidence: float | None = None) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=event_type,
        component=component,
        timestamp="2026-01-01T00:00:00+00:00",
        duration=0.1,
        confidence=confidence,
    )


def test_execution_event_is_frozen():
    event = _event("planner", "planner_started")
    try:
        event.component = "retrieval"  # type: ignore[misc]
        assert False, "ExecutionEvent should be immutable"
    except AttributeError:
        pass


def test_execution_trace_is_frozen():
    trace = ExecutionTrace(execution_id="e1", query="q")
    try:
        trace.query = "other"  # type: ignore[misc]
        assert False, "ExecutionTrace should be immutable"
    except AttributeError:
        pass


def test_event_count_reflects_appended_events():
    trace = ExecutionTrace(execution_id="e1", query="q", events=[_event("planner", "planner_started"), _event("planner", "planner_finished")])
    assert trace.event_count() == 2


def test_events_for_component_filters_correctly():
    trace = ExecutionTrace(
        execution_id="e1", query="q",
        events=[_event("planner", "planner_started"), _event("retrieval", "retrieval_started"), _event("planner", "planner_finished")],
    )
    assert len(trace.events_for_component("planner")) == 2
    assert len(trace.events_for_component("retrieval")) == 1
    assert trace.events_for_component("nonexistent") == []


def test_components_preserves_first_seen_order_without_duplicates():
    trace = ExecutionTrace(
        execution_id="e1", query="q",
        events=[
            _event("planner", "planner_started"), _event("planner", "planner_finished"),
            _event("retrieval", "retrieval_started"), _event("retrieval", "retrieval_finished"),
        ],
    )
    assert trace.components() == ["planner", "retrieval"]


def test_subsystem_trace_slots_default_to_none():
    trace = ExecutionTrace(execution_id="e1", query="q")
    assert trace.compression_trace is None
    assert trace.vision_trace is None
    assert trace.prediction_trace is None
    assert trace.graph_trace is None


def test_trace_carries_latencies_memory_confidence_timestamps():
    trace = ExecutionTrace(
        execution_id="e1", query="q",
        latencies={"planner": 0.01}, memory={"planner": 1024},
        confidence=0.8, timestamps={"started_at": "t0", "completed_at": "t1"},
    )
    assert trace.latencies["planner"] == 0.01
    assert trace.memory["planner"] == 1024
    assert trace.confidence == 0.8
    assert trace.timestamps["started_at"] == "t0"
