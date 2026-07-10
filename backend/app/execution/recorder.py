"""Execution Recorder (O3.3 Component 3).

A reusable helper: pipeline stages never manually update metrics themselves —
`app/orchestrator/graph.py`'s instrumentation wrapper (Component 4) is the
only caller, and it calls `ExecutionRecorder` around every stage. No agent's
own code knows this class exists ("no manual bookkeeping inside agents").

Deliberately free of hidden/mutable global state ("no hidden globals"):
every method takes what it needs and returns a new immutable value, matching
the append-only `{**state, ...}` style the rest of the orchestrator already
uses for `AgentState`.
"""

import datetime as dt
import uuid

from app.execution.models import ExecutionEvent, ExecutionTrace


def current_memory_bytes() -> int | None:
    """Real process RSS via psutil (an existing declared dependency).

    Returns None — never a fabricated number — if psutil or the underlying
    platform call is unavailable, matching this project's standing
    "gracefully degrade optional dependencies" constraint.
    """
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        return None


class ExecutionRecorder:
    @staticmethod
    def new_execution_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now() -> str:
        return dt.datetime.now(dt.timezone.utc).isoformat()

    @staticmethod
    def record_started(component: str, query: str) -> ExecutionEvent:
        return ExecutionEvent(
            event_type=f"{component}_started",
            component=component,
            timestamp=ExecutionRecorder.now(),
            duration=0.0,
            inputs={"query": query},
            memory_bytes=current_memory_bytes(),
        )

    @staticmethod
    def record_finished(
        component: str,
        query: str,
        duration: float,
        produced_keys: list[str],
        confidence: float | None,
        memory_before: int | None,
    ) -> ExecutionEvent:
        memory_after = current_memory_bytes()
        memory_delta = (
            memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None
        )
        return ExecutionEvent(
            event_type=f"{component}_finished",
            component=component,
            timestamp=ExecutionRecorder.now(),
            duration=duration,
            inputs={"query": query},
            outputs={"produced_keys": produced_keys},
            confidence=confidence,
            latency=duration,
            memory_bytes=memory_delta,
        )

    @staticmethod
    def finalize(
        execution_id: str,
        query: str,
        events: list[ExecutionEvent],
        latencies: dict[str, float],
        confidence: float,
        started_at: str,
        completed_at: str,
        subsystem_traces: dict[str, dict | None] | None = None,
    ) -> ExecutionTrace:
        subsystem_traces = subsystem_traces or {}
        memory = {
            event.component: event.memory_bytes
            for event in events
            if event.event_type.endswith("_finished") and event.memory_bytes is not None
        }
        return ExecutionTrace(
            execution_id=execution_id,
            query=query,
            events=events,
            latencies=latencies,
            memory=memory,
            confidence=confidence,
            timestamps={"started_at": started_at, "completed_at": completed_at},
            planner_trace=subsystem_traces.get("planner"),
            retrieval_trace=subsystem_traces.get("retrieval"),
            ranking_trace=subsystem_traces.get("ranking"),
            compression_trace=subsystem_traces.get("compression"),
            claim_extraction_trace=subsystem_traces.get("claim_extraction"),
            verification_trace=subsystem_traces.get("verification"),
            nli_trace=subsystem_traces.get("nli"),
            recommendation_trace=subsystem_traces.get("recommendation"),
            faithfulness_trace=subsystem_traces.get("faithfulness"),
            reflection_trace=subsystem_traces.get("reflection"),
            vision_trace=subsystem_traces.get("vision"),
            prediction_trace=subsystem_traces.get("prediction"),
            graph_trace=subsystem_traces.get("graph"),
            cache_trace=subsystem_traces.get("cache"),
        )
