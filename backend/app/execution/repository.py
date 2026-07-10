"""Execution Repository (O3.3 Component 6).

Persists `ExecutionTrace` (+ its `ExecutionEvent`s) and `FailureEvent`s.
Same SQLite-via-async-SQLAlchemy pattern as
`app/research/experiments/repository.py` / `app/error_analysis/repository.py`
(dedicated tables, dedicated repository, no shared mega-table, no
duplicated storage of what's already in those other tables) so a future
swap to a real analytics DB is a drop-in behind
`ExecutionRepositoryProtocol`.
"""

import datetime as dt
import json
import uuid
from typing import Protocol

from sqlalchemy import Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.error_analysis.models import FailureCategory, FailureSeverity
from app.execution.failure_event import FailureEvent
from app.execution.models import ExecutionEvent, ExecutionTrace


class Base(DeclarativeBase):
    pass


class ExecutionTraceRecord(Base):
    __tablename__ = "execution_traces"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(String)
    latencies_json: Mapped[str] = mapped_column(String)
    memory_json: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    started_at: Mapped[str] = mapped_column(String)
    completed_at: Mapped[str] = mapped_column(String)
    subsystem_traces_json: Mapped[str] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String)


class ExecutionEventRecord(Base):
    __tablename__ = "execution_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    execution_id: Mapped[str] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String)
    component: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)
    duration: Mapped[float] = mapped_column(Float)
    inputs_json: Mapped[str] = mapped_column(String)
    outputs_json: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[str] = mapped_column(String)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency: Mapped[float] = mapped_column(Float)
    memory_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)


class FailureEventRecord(Base):
    __tablename__ = "failure_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    execution_id: Mapped[str] = mapped_column(String)
    component: Mapped[str] = mapped_column(String)
    failure_type: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    root_cause: Mapped[str] = mapped_column(String)
    evidence_ids_json: Mapped[str] = mapped_column(String)
    claim_ids_json: Mapped[str] = mapped_column(String)
    graph_nodes_json: Mapped[str] = mapped_column(String)
    metadata_json: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)


_SUBSYSTEM_TRACE_FIELDS = [
    "planner_trace", "retrieval_trace", "ranking_trace", "compression_trace",
    "claim_extraction_trace", "verification_trace", "nli_trace", "recommendation_trace",
    "faithfulness_trace", "reflection_trace", "vision_trace", "prediction_trace",
    "graph_trace", "cache_trace",
]


def _event_to_record(execution_id: str, event: ExecutionEvent) -> ExecutionEventRecord:
    return ExecutionEventRecord(
        id=str(uuid.uuid4()),
        execution_id=execution_id,
        event_type=event.event_type,
        component=event.component,
        timestamp=event.timestamp,
        duration=event.duration,
        inputs_json=json.dumps(event.inputs),
        outputs_json=json.dumps(event.outputs),
        metadata_json=json.dumps(event.metadata),
        confidence=event.confidence,
        latency=event.latency,
        memory_bytes=event.memory_bytes,
    )


def _record_to_event(record: ExecutionEventRecord) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=record.event_type,
        component=record.component,
        timestamp=record.timestamp,
        duration=record.duration,
        inputs=json.loads(record.inputs_json),
        outputs=json.loads(record.outputs_json),
        metadata=json.loads(record.metadata_json),
        confidence=record.confidence,
        latency=record.latency,
        memory_bytes=record.memory_bytes,
    )


def _record_to_failure_event(record: FailureEventRecord) -> FailureEvent:
    return FailureEvent(
        id=record.id,
        execution_id=record.execution_id,
        component=record.component,
        failure_type=FailureCategory(record.failure_type),
        severity=FailureSeverity(record.severity),
        confidence=record.confidence,
        root_cause=record.root_cause,
        evidence_ids=json.loads(record.evidence_ids_json),
        claim_ids=json.loads(record.claim_ids_json),
        graph_nodes=json.loads(record.graph_nodes_json),
        metadata=json.loads(record.metadata_json),
    )


class ExecutionRepositoryProtocol(Protocol):
    async def init_models(self) -> None: ...
    async def save_trace(self, trace: ExecutionTrace) -> str: ...
    async def get_trace(self, execution_id: str) -> ExecutionTrace | None: ...
    async def list_traces(self, limit: int = 50) -> list[ExecutionTrace]: ...
    async def save_failure_event(self, event: FailureEvent) -> str: ...
    async def list_failure_events(self, execution_id: str | None = None) -> list[FailureEvent]: ...
    async def statistics(self) -> dict: ...


class SqlExecutionRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save_trace(self, trace: ExecutionTrace) -> str:
        subsystem_traces = {field_name: getattr(trace, field_name) for field_name in _SUBSYSTEM_TRACE_FIELDS}
        record = ExecutionTraceRecord(
            id=trace.execution_id,
            query=trace.query,
            latencies_json=json.dumps(trace.latencies),
            memory_json=json.dumps(trace.memory),
            confidence=trace.confidence,
            started_at=trace.timestamps.get("started_at", ""),
            completed_at=trace.timestamps.get("completed_at", ""),
            subsystem_traces_json=json.dumps(subsystem_traces),
            created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        async with self._session_factory() as session:
            session.add(record)
            for event in trace.events:
                session.add(_event_to_record(trace.execution_id, event))
            await session.commit()
        return trace.execution_id

    async def get_trace(self, execution_id: str) -> ExecutionTrace | None:
        async with self._session_factory() as session:
            trace_record = await session.get(ExecutionTraceRecord, execution_id)
            if trace_record is None:
                return None
            event_result = await session.execute(
                select(ExecutionEventRecord)
                .where(ExecutionEventRecord.execution_id == execution_id)
                .order_by(ExecutionEventRecord.timestamp)
            )
            events = [_record_to_event(r) for r in event_result.scalars().all()]
            return self._trace_record_to_trace(trace_record, events)

    async def list_traces(self, limit: int = 50) -> list[ExecutionTrace]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ExecutionTraceRecord).order_by(ExecutionTraceRecord.created_at.desc()).limit(limit)
            )
            traces = []
            for trace_record in result.scalars().all():
                event_result = await session.execute(
                    select(ExecutionEventRecord)
                    .where(ExecutionEventRecord.execution_id == trace_record.id)
                    .order_by(ExecutionEventRecord.timestamp)
                )
                events = [_record_to_event(r) for r in event_result.scalars().all()]
                traces.append(self._trace_record_to_trace(trace_record, events))
            return traces

    async def save_failure_event(self, event: FailureEvent) -> str:
        record = FailureEventRecord(
            id=event.id,
            execution_id=event.execution_id,
            component=event.component,
            failure_type=event.failure_type.value,
            severity=event.severity.value,
            confidence=event.confidence,
            root_cause=event.root_cause,
            evidence_ids_json=json.dumps(event.evidence_ids),
            claim_ids_json=json.dumps(event.claim_ids),
            graph_nodes_json=json.dumps(event.graph_nodes),
            metadata_json=json.dumps(event.metadata),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
        return event.id

    async def list_failure_events(self, execution_id: str | None = None) -> list[FailureEvent]:
        async with self._session_factory() as session:
            query = select(FailureEventRecord).order_by(FailureEventRecord.timestamp.desc())
            if execution_id is not None:
                query = query.where(FailureEventRecord.execution_id == execution_id)
            result = await session.execute(query)
            return [_record_to_failure_event(r) for r in result.scalars().all()]

    async def statistics(self) -> dict:
        async with self._session_factory() as session:
            trace_count = await session.scalar(select(func.count()).select_from(ExecutionTraceRecord)) or 0
            event_count = await session.scalar(select(func.count()).select_from(ExecutionEventRecord)) or 0
            avg_confidence = await session.scalar(select(func.avg(ExecutionTraceRecord.confidence)))
            component_avg_result = await session.execute(
                select(ExecutionEventRecord.component, func.avg(ExecutionEventRecord.duration))
                .where(ExecutionEventRecord.event_type.like("%_finished"))
                .group_by(ExecutionEventRecord.component)
            )
            average_latency_by_component = {
                component: round(avg_duration, 4) for component, avg_duration in component_avg_result.all()
            }
            return {
                "trace_count": trace_count,
                "event_count": event_count,
                "average_confidence": round(avg_confidence, 4) if avg_confidence is not None else 0.0,
                "average_latency_by_component": average_latency_by_component,
            }

    @staticmethod
    def _trace_record_to_trace(record: ExecutionTraceRecord, events: list[ExecutionEvent]) -> ExecutionTrace:
        subsystem_traces = json.loads(record.subsystem_traces_json)
        return ExecutionTrace(
            execution_id=record.id,
            query=record.query,
            events=events,
            latencies=json.loads(record.latencies_json),
            memory=json.loads(record.memory_json),
            confidence=record.confidence,
            timestamps={"started_at": record.started_at, "completed_at": record.completed_at},
            **{field_name: subsystem_traces.get(field_name) for field_name in _SUBSYSTEM_TRACE_FIELDS},
        )
