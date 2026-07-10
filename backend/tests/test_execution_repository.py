from app.error_analysis.models import DetectedFailure, FailureCategory, FailureSeverity
from app.execution.failure_event import FailureEvent
from app.execution.models import ExecutionEvent, ExecutionTrace
from app.execution.repository import SqlExecutionRepository


async def _repository(tmp_path) -> SqlExecutionRepository:
    db_path = tmp_path / "execution_test.db"
    repository = SqlExecutionRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


def _trace(execution_id: str = "e1", query: str = "why did P-101 fail?") -> ExecutionTrace:
    events = [
        ExecutionEvent(event_type="planning_started", component="planning", timestamp="t0", duration=0.0),
        ExecutionEvent(event_type="planning_finished", component="planning", timestamp="t1", duration=0.05, confidence=0.6, latency=0.05, memory_bytes=1024),
    ]
    return ExecutionTrace(
        execution_id=execution_id, query=query, events=events,
        latencies={"planning": 0.05}, memory={"planning": 1024}, confidence=0.6,
        timestamps={"started_at": "t0", "completed_at": "t1"},
        planner_trace={"nodes": 1}, faithfulness_trace={"score": 0.8},
    )


async def test_save_and_get_trace_round_trips_events_and_subsystem_traces(tmp_path):
    repository = await _repository(tmp_path)

    execution_id = await repository.save_trace(_trace())
    assert execution_id == "e1"

    trace = await repository.get_trace("e1")
    assert trace is not None
    assert trace.query == "why did P-101 fail?"
    assert trace.confidence == 0.6
    assert trace.planner_trace == {"nodes": 1}
    assert trace.faithfulness_trace == {"score": 0.8}
    assert trace.compression_trace is None
    assert len(trace.events) == 2
    assert trace.events[0].event_type == "planning_started"
    assert trace.events[1].memory_bytes == 1024


async def test_get_trace_returns_none_for_unknown_execution_id(tmp_path):
    repository = await _repository(tmp_path)
    assert await repository.get_trace("nonexistent") is None


async def test_list_traces_orders_newest_first(tmp_path):
    repository = await _repository(tmp_path)
    await repository.save_trace(_trace("e1"))
    await repository.save_trace(_trace("e2"))

    traces = await repository.list_traces()
    assert {t.execution_id for t in traces} == {"e1", "e2"}


async def test_save_and_list_failure_events(tmp_path):
    repository = await _repository(tmp_path)
    failure = DetectedFailure(
        category=FailureCategory.RETRIEVAL_FAILURE, root_cause="no evidence", affected_components=["HybridRetriever"],
        confidence=0.9, suggested_improvement="check ingestion", severity=FailureSeverity.CRITICAL,
    )
    event = FailureEvent.from_detected_failure("e1", failure)

    event_id = await repository.save_failure_event(event)
    assert event_id == event.id

    all_events = await repository.list_failure_events()
    assert len(all_events) == 1
    assert all_events[0].execution_id == "e1"
    assert all_events[0].severity == FailureSeverity.CRITICAL

    scoped_events = await repository.list_failure_events(execution_id="e1")
    assert len(scoped_events) == 1

    other_scoped = await repository.list_failure_events(execution_id="e2")
    assert other_scoped == []


async def test_statistics_aggregates_across_stored_traces(tmp_path):
    repository = await _repository(tmp_path)
    await repository.save_trace(_trace("e1"))
    await repository.save_trace(_trace("e2"))

    stats = await repository.statistics()
    assert stats["trace_count"] == 2
    assert stats["event_count"] == 4
    assert stats["average_confidence"] == 0.6
    assert stats["average_latency_by_component"]["planning"] == 0.05


async def test_statistics_on_empty_repository_returns_zeros(tmp_path):
    repository = await _repository(tmp_path)
    stats = await repository.statistics()
    assert stats["trace_count"] == 0
    assert stats["event_count"] == 0
    assert stats["average_confidence"] == 0.0
    assert stats["average_latency_by_component"] == {}
