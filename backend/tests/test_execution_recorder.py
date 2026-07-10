from app.execution.recorder import ExecutionRecorder, current_memory_bytes


def test_new_execution_id_is_unique():
    assert ExecutionRecorder.new_execution_id() != ExecutionRecorder.new_execution_id()


def test_record_started_produces_a_started_event_with_query_input():
    event = ExecutionRecorder.record_started("planner", "why did P-101 fail?")

    assert event.event_type == "planner_started"
    assert event.component == "planner"
    assert event.duration == 0.0
    assert event.inputs["query"] == "why did P-101 fail?"


def test_record_finished_produces_a_finished_event_with_duration_and_outputs():
    event = ExecutionRecorder.record_finished(
        "retrieval", "q", duration=0.25, produced_keys=["evidence_bundle"], confidence=0.7, memory_before=None,
    )

    assert event.event_type == "retrieval_finished"
    assert event.duration == 0.25
    assert event.latency == 0.25
    assert event.outputs["produced_keys"] == ["evidence_bundle"]
    assert event.confidence == 0.7


def test_record_finished_computes_real_memory_delta_when_before_known():
    event = ExecutionRecorder.record_finished(
        "planner", "q", duration=0.01, produced_keys=[], confidence=None, memory_before=0,
    )
    # memory_after (real RSS) minus memory_before=0 should equal memory_after,
    # which is a large positive number for any running Python process — the
    # point is it's a real computed delta, not a fabricated placeholder.
    if current_memory_bytes() is not None:
        assert event.memory_bytes is not None
        assert event.memory_bytes > 0


def test_record_finished_memory_bytes_none_when_before_unknown():
    event = ExecutionRecorder.record_finished(
        "planner", "q", duration=0.01, produced_keys=[], confidence=None, memory_before=None,
    )
    assert event.memory_bytes is None


def test_finalize_builds_trace_with_memory_derived_only_from_finished_events():
    started = ExecutionRecorder.record_started("planner", "q")
    finished = ExecutionRecorder.record_finished("planner", "q", 0.1, ["plan"], 0.5, memory_before=0)

    trace = ExecutionRecorder.finalize(
        execution_id="e1", query="q", events=[started, finished],
        latencies={"planner": 0.1}, confidence=0.5,
        started_at="t0", completed_at="t1",
    )

    assert trace.execution_id == "e1"
    assert trace.latencies == {"planner": 0.1}
    assert trace.timestamps == {"started_at": "t0", "completed_at": "t1"}
    assert "planner" in trace.memory or current_memory_bytes() is None


def test_finalize_maps_subsystem_traces_into_named_slots():
    trace = ExecutionRecorder.finalize(
        execution_id="e1", query="q", events=[], latencies={}, confidence=0.0,
        started_at="t0", completed_at="t1",
        subsystem_traces={"planner": {"nodes": 2}, "faithfulness": {"score": 0.9}},
    )

    assert trace.planner_trace == {"nodes": 2}
    assert trace.faithfulness_trace == {"score": 0.9}
    assert trace.compression_trace is None
    assert trace.vision_trace is None
