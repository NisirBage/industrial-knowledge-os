"""Failure Taxonomy persistence (O3 Component 5).

Every failed query's `DetectedFailure`(s) get persisted here — a structured,
queryable record (failure type, root cause, affected components,
confidence, suggested improvement) rather than only living in one chat
response. Same SQLite-via-async-SQLAlchemy pattern as
`app/research/experiments/repository.py` (one dedicated table, one
dedicated repository, no shared mega-table) so a future swap to a real
analytics DB is a drop-in behind `FailureTaxonomyRepositoryProtocol`.
"""

import datetime as dt
import json
import uuid
from typing import Protocol

from sqlalchemy import Float, Integer, String, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.error_analysis.models import DetectedFailure, FailureCategory, FailureSeverity


class Base(DeclarativeBase):
    pass


class FailureRecord(Base):
    __tablename__ = "failure_taxonomy"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    root_cause: Mapped[str] = mapped_column(String)
    affected_components_json: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    suggested_improvement: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String, default=FailureSeverity.MEDIUM.value)
    frequency: Mapped[int] = mapped_column(Integer, default=1)


def _record_to_failure(record: FailureRecord) -> DetectedFailure:
    return DetectedFailure(
        category=FailureCategory(record.category),
        root_cause=record.root_cause,
        affected_components=json.loads(record.affected_components_json),
        confidence=record.confidence,
        suggested_improvement=record.suggested_improvement,
        severity=FailureSeverity(record.severity),
        frequency=record.frequency,
    )


class FailureTaxonomyRepositoryProtocol(Protocol):
    async def init_models(self) -> None: ...
    async def record(self, query: str, failure: DetectedFailure) -> str: ...
    async def list_by_category(self, category: FailureCategory | None = None) -> list[DetectedFailure]: ...
    async def category_counts(self) -> dict[str, int]: ...


class SqlFailureTaxonomyRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def record(self, query: str, failure: DetectedFailure) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            existing_count = await session.scalar(
                select(func.count()).select_from(FailureRecord).where(FailureRecord.category == failure.category.value)
            )
            record = FailureRecord(
                id=record_id,
                query=query,
                category=failure.category.value,
                root_cause=failure.root_cause,
                affected_components_json=json.dumps(failure.affected_components),
                confidence=failure.confidence,
                suggested_improvement=failure.suggested_improvement,
                timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
                severity=failure.severity.value,
                # Real historical frequency for this category, computed at
                # persistence time — a freshly-detected DetectedFailure has
                # no way to know this itself (see models.py).
                frequency=(existing_count or 0) + 1,
            )
            session.add(record)
            await session.commit()
        return record_id

    async def list_by_category(self, category: FailureCategory | None = None) -> list[DetectedFailure]:
        async with self._session_factory() as session:
            query = select(FailureRecord).order_by(FailureRecord.timestamp.desc())
            if category is not None:
                query = query.where(FailureRecord.category == category.value)
            result = await session.execute(query)
            return [_record_to_failure(r) for r in result.scalars().all()]

    async def category_counts(self) -> dict[str, int]:
        async with self._session_factory() as session:
            result = await session.execute(select(FailureRecord.category))
            counts: dict[str, int] = {}
            for (category,) in result.all():
                counts[category] = counts.get(category, 0) + 1
            return counts
