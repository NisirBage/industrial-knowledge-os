"""Timeline Reconstruction (O3.3 Component 7).

Given one `ExecutionTrace`, reconstruct real execution order, per-component
timing spans, confidence evolution, and a DAG — entirely from the trace's
own recorded events, never re-deriving timing a stage already computed.

The current orchestrator (`app/orchestrator/graph.py`) runs every stage
strictly sequentially, so overlap detection here will honestly report no
parallel branches for it today — this module doesn't fabricate parallelism
that doesn't exist; it becomes meaningful the moment a future stage
genuinely runs concurrently and emits overlapping Started/Finished events.
"""

import datetime as dt
from dataclasses import dataclass, field

from app.execution.models import ExecutionTrace


@dataclass(frozen=True)
class TimelineSpan:
    component: str
    started_at: str
    finished_at: str
    duration_seconds: float
    confidence: float | None = None
    memory_bytes: int | None = None


@dataclass(frozen=True)
class ExecutionDAG:
    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ReconstructedTimeline:
    spans: list[TimelineSpan]
    component_order: list[str]
    parallel_branches: list[tuple[str, str]]
    cache_hits: list[str]
    confidence_evolution: list[tuple[str, float]]
    dag: ExecutionDAG


def _parse(timestamp: str) -> dt.datetime:
    return dt.datetime.fromisoformat(timestamp)


def reconstruct_timeline(trace: ExecutionTrace) -> ReconstructedTimeline:
    spans_sorted = sorted(_build_spans(trace), key=lambda s: _parse(s.started_at))

    return ReconstructedTimeline(
        spans=spans_sorted,
        component_order=[s.component for s in spans_sorted],
        parallel_branches=_detect_overlaps(spans_sorted),
        cache_hits=_extract_cache_hits(trace),
        confidence_evolution=[(s.component, s.confidence) for s in spans_sorted if s.confidence is not None],
        dag=_build_dag(spans_sorted),
    )


def _build_spans(trace: ExecutionTrace) -> list[TimelineSpan]:
    spans = []
    for component in trace.components():
        component_events = trace.events_for_component(component)
        started = next((e for e in component_events if e.event_type.endswith("_started")), None)
        finished = next((e for e in component_events if e.event_type.endswith("_finished")), None)
        if started is None or finished is None:
            continue
        spans.append(
            TimelineSpan(
                component=component,
                started_at=started.timestamp,
                finished_at=finished.timestamp,
                duration_seconds=finished.duration,
                confidence=finished.confidence,
                memory_bytes=finished.memory_bytes,
            )
        )
    return spans


def _detect_overlaps(spans_sorted: list[TimelineSpan]) -> list[tuple[str, str]]:
    overlaps = []
    for i in range(len(spans_sorted)):
        a_start, a_end = _parse(spans_sorted[i].started_at), _parse(spans_sorted[i].finished_at)
        for j in range(i + 1, len(spans_sorted)):
            b_start, b_end = _parse(spans_sorted[j].started_at), _parse(spans_sorted[j].finished_at)
            if a_start < b_end and b_start < a_end:
                overlaps.append((spans_sorted[i].component, spans_sorted[j].component))
    return overlaps


def _extract_cache_hits(trace: ExecutionTrace) -> list[str]:
    return [e.component for e in trace.events if e.metadata.get("cache_hit") is True]


def _build_dag(spans_sorted: list[TimelineSpan]) -> ExecutionDAG:
    nodes = [s.component for s in spans_sorted]
    edges = [(spans_sorted[i].component, spans_sorted[i + 1].component) for i in range(len(spans_sorted) - 1)]
    return ExecutionDAG(nodes=nodes, edges=edges)
