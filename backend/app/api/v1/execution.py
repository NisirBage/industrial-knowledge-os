"""Execution Trace API (O3.3): read-side surface over the
`SqlExecutionRepository` populated by every chat turn's `ExecutionTrace` —
list/lookup/timeline reconstruction/aggregate statistics/failure events.
Entirely read-only; nothing here is on the chat request path itself (see
`ChatService._persist_execution_trace` for the write side)."""

from fastapi import APIRouter, HTTPException, Request

from app.execution.models import ExecutionTrace
from app.execution.timeline import ReconstructedTimeline, reconstruct_timeline
from app.schemas.execution import (
    ExecutionDAGOut,
    ExecutionEventOut,
    ExecutionStatisticsOut,
    ExecutionTraceOut,
    FailureEventOut,
    ReconstructedTimelineOut,
    TimelineSpanOut,
)

router = APIRouter(prefix="/executions", tags=["execution"])


def _trace_to_out(trace: ExecutionTrace) -> ExecutionTraceOut:
    return ExecutionTraceOut(
        execution_id=trace.execution_id,
        query=trace.query,
        events=[
            ExecutionEventOut(
                event_type=e.event_type, component=e.component, timestamp=e.timestamp, duration=e.duration,
                inputs=e.inputs, outputs=e.outputs, metadata=e.metadata, confidence=e.confidence,
                latency=e.latency, memory_bytes=e.memory_bytes,
            )
            for e in trace.events
        ],
        latencies=trace.latencies,
        memory=trace.memory,
        confidence=trace.confidence,
        timestamps=trace.timestamps,
        planner_trace=trace.planner_trace,
        retrieval_trace=trace.retrieval_trace,
        ranking_trace=trace.ranking_trace,
        compression_trace=trace.compression_trace,
        claim_extraction_trace=trace.claim_extraction_trace,
        verification_trace=trace.verification_trace,
        nli_trace=trace.nli_trace,
        recommendation_trace=trace.recommendation_trace,
        faithfulness_trace=trace.faithfulness_trace,
        reflection_trace=trace.reflection_trace,
        vision_trace=trace.vision_trace,
        prediction_trace=trace.prediction_trace,
        graph_trace=trace.graph_trace,
        cache_trace=trace.cache_trace,
    )


def _timeline_to_out(timeline: ReconstructedTimeline) -> ReconstructedTimelineOut:
    return ReconstructedTimelineOut(
        spans=[
            TimelineSpanOut(
                component=s.component, started_at=s.started_at, finished_at=s.finished_at,
                duration_seconds=s.duration_seconds, confidence=s.confidence, memory_bytes=s.memory_bytes,
            )
            for s in timeline.spans
        ],
        component_order=timeline.component_order,
        parallel_branches=[list(pair) for pair in timeline.parallel_branches],
        cache_hits=timeline.cache_hits,
        confidence_evolution=[list(pair) for pair in timeline.confidence_evolution],
        dag=ExecutionDAGOut(nodes=timeline.dag.nodes, edges=[list(edge) for edge in timeline.dag.edges]),
    )


@router.get("/statistics", response_model=ExecutionStatisticsOut)
async def get_execution_statistics(request: Request) -> ExecutionStatisticsOut:
    repository = request.app.state.execution_repository
    stats = await repository.statistics()
    return ExecutionStatisticsOut(**stats)


@router.get("", response_model=list[ExecutionTraceOut])
async def list_executions(request: Request, limit: int = 50) -> list[ExecutionTraceOut]:
    repository = request.app.state.execution_repository
    traces = await repository.list_traces(limit=limit)
    return [_trace_to_out(t) for t in traces]


@router.get("/{execution_id}", response_model=ExecutionTraceOut)
async def get_execution(execution_id: str, request: Request) -> ExecutionTraceOut:
    repository = request.app.state.execution_repository
    trace = await repository.get_trace(execution_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No execution trace with id {execution_id!r}")
    return _trace_to_out(trace)


@router.get("/{execution_id}/timeline", response_model=ReconstructedTimelineOut)
async def get_execution_timeline(execution_id: str, request: Request) -> ReconstructedTimelineOut:
    repository = request.app.state.execution_repository
    trace = await repository.get_trace(execution_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No execution trace with id {execution_id!r}")
    return _timeline_to_out(reconstruct_timeline(trace))


@router.get("/failures/list", response_model=list[FailureEventOut])
async def list_failure_events(request: Request, execution_id: str | None = None) -> list[FailureEventOut]:
    repository = request.app.state.execution_repository
    events = await repository.list_failure_events(execution_id=execution_id)
    return [
        FailureEventOut(
            id=e.id, execution_id=e.execution_id, component=e.component, failure_type=e.failure_type.value,
            severity=e.severity.value, confidence=e.confidence, root_cause=e.root_cause,
            evidence_ids=e.evidence_ids, claim_ids=e.claim_ids, graph_nodes=e.graph_nodes, metadata=e.metadata,
        )
        for e in events
    ]
