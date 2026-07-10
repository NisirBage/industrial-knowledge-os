"""FailurePattern persistence (O3.4 Component 1).

Own table, own repository — same SQLite-via-async-SQLAlchemy pattern as
`app/error_analysis/repository.py` / `app/research/experiments/repository.py`
(no shared mega-table, no duplicated storage of what `SqlExecutionRepository`
already owns)."""

import json
import uuid
from typing import Protocol

from sqlalchemy import Float, Integer, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.error_analysis.models import FailureSeverity
from app.failure_intelligence.models import FailurePattern


class Base(DeclarativeBase):
    pass


class FailurePatternRecord(Base):
    __tablename__ = "failure_patterns"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    pattern_id: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String)
    support: Mapped[int] = mapped_column(Integer)
    frequency: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String)
    affected_components_json: Mapped[str] = mapped_column(String)
    example_execution_ids_json: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String)


def _record_to_pattern(record: FailurePatternRecord) -> FailurePattern:
    return FailurePattern(
        pattern_id=record.pattern_id,
        category=record.category,
        support=record.support,
        frequency=record.frequency,
        severity=FailureSeverity(record.severity),
        affected_components=json.loads(record.affected_components_json),
        example_execution_ids=json.loads(record.example_execution_ids_json),
        confidence=record.confidence,
        description=record.description,
    )


class FailurePatternRepositoryProtocol(Protocol):
    async def init_models(self) -> None: ...
    async def save(self, pattern: FailurePattern) -> str: ...
    async def save_all(self, patterns: list[FailurePattern]) -> list[str]: ...
    async def list_all(self) -> list[FailurePattern]: ...
    async def get(self, pattern_id: str) -> FailurePattern | None: ...


class SqlFailurePatternRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save(self, pattern: FailurePattern) -> str:
        record = FailurePatternRecord(
            id=str(uuid.uuid4()),
            pattern_id=pattern.pattern_id,
            category=pattern.category,
            support=pattern.support,
            frequency=pattern.frequency,
            severity=pattern.severity.value,
            affected_components_json=json.dumps(pattern.affected_components),
            example_execution_ids_json=json.dumps(pattern.example_execution_ids),
            confidence=pattern.confidence,
            description=pattern.description,
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
        return pattern.pattern_id

    async def save_all(self, patterns: list[FailurePattern]) -> list[str]:
        return [await self.save(pattern) for pattern in patterns]

    async def list_all(self) -> list[FailurePattern]:
        async with self._session_factory() as session:
            result = await session.execute(select(FailurePatternRecord))
            return [_record_to_pattern(r) for r in result.scalars().all()]

    async def get(self, pattern_id: str) -> FailurePattern | None:
        async with self._session_factory() as session:
            result = await session.execute(select(FailurePatternRecord).where(FailurePatternRecord.pattern_id == pattern_id))
            record = result.scalars().first()
            return _record_to_pattern(record) if record is not None else None
