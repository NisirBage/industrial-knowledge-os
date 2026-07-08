"""SQLite-backed Experiment storage (M5 Component 1).

Deliberately the only place SQLAlchemy appears for experiments — everything
else works with the plain Experiment dataclass. Swapping to MLflow/W&B/Neptune
later means writing one new class implementing ExperimentTrackerProtocol;
nothing else in the codebase changes, since ExperimentManager (manager.py)
and every caller only ever see the protocol.

Versioned-component lists and metrics are stored as JSON text columns —
SQLite has no native list/dict column type, and this avoids pulling in
SQLAlchemy's JSON type quirks across dialects for a field that's opaque to
SQL anyway (nothing queries inside `model_versions`).
"""

import dataclasses
import datetime as dt
import json
import uuid
from typing import Protocol

from sqlalchemy import Float, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.research.experiments.models import Experiment, ExperimentStatus
from app.research.model_registry import ComponentKind, ComponentVersion


class Base(DeclarativeBase):
    pass


class ExperimentRecord(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    configuration_json: Mapped[str] = mapped_column(String)
    model_versions_json: Mapped[str] = mapped_column(String)
    retriever_versions_json: Mapped[str] = mapped_column(String)
    graph_versions_json: Mapped[str] = mapped_column(String)
    reranker: Mapped[str] = mapped_column(String)
    ontology_version: Mapped[str] = mapped_column(String)
    dataset: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[str] = mapped_column(String, default="{}")
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    random_seed: Mapped[int | None] = mapped_column(nullable=True)
    git_commit: Mapped[str | None] = mapped_column(String, nullable=True)
    error: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at_epoch: Mapped[float] = mapped_column(Float)


def _versions_to_json(versions: list[ComponentVersion]) -> str:
    return json.dumps([dataclasses.asdict(v) for v in versions])


def _versions_from_json(raw: str) -> list[ComponentVersion]:
    # ComponentKind is a str subclass, so json.dumps silently accepts it as
    # a plain string on the way in — json.loads has no way to know it should
    # come back out as an enum, so that conversion has to happen explicitly
    # here rather than via a bare ComponentVersion(**item).
    return [ComponentVersion(**{**item, "kind": ComponentKind(item["kind"])}) for item in json.loads(raw)]


def _record_to_experiment(record: ExperimentRecord) -> Experiment:
    return Experiment(
        experiment_id=record.experiment_id,
        name=record.name,
        description=record.description,
        configuration=json.loads(record.configuration_json),
        model_versions=_versions_from_json(record.model_versions_json),
        retriever_versions=_versions_from_json(record.retriever_versions_json),
        graph_versions=_versions_from_json(record.graph_versions_json),
        reranker=record.reranker,
        ontology_version=record.ontology_version,
        dataset=record.dataset,
        timestamp=record.timestamp,
        status=ExperimentStatus(record.status),
        metrics=json.loads(record.metrics_json),
        duration=record.duration,
        random_seed=record.random_seed,
        git_commit=record.git_commit,
        error=record.error,
    )


class ExperimentTrackerProtocol(Protocol):
    async def init_models(self) -> None: ...

    async def create(
        self,
        name: str,
        description: str,
        configuration: dict[str, str],
        model_versions: list[ComponentVersion],
        retriever_versions: list[ComponentVersion],
        graph_versions: list[ComponentVersion],
        reranker: str,
        ontology_version: str,
        dataset: str,
        random_seed: int | None,
        git_commit: str | None,
    ) -> Experiment: ...

    async def complete(self, experiment_id: str, metrics: dict[str, float]) -> Experiment: ...
    async def fail(self, experiment_id: str, error: str) -> Experiment: ...
    async def get(self, experiment_id: str) -> Experiment | None: ...
    async def list(self) -> list[Experiment]: ...


class SqlExperimentTracker:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create(
        self,
        name: str,
        description: str,
        configuration: dict[str, str],
        model_versions: list[ComponentVersion],
        retriever_versions: list[ComponentVersion],
        graph_versions: list[ComponentVersion],
        reranker: str,
        ontology_version: str,
        dataset: str,
        random_seed: int | None = None,
        git_commit: str | None = None,
    ) -> Experiment:
        experiment_id = str(uuid.uuid4())
        now = dt.datetime.now(dt.timezone.utc)
        record = ExperimentRecord(
            experiment_id=experiment_id,
            name=name,
            description=description,
            configuration_json=json.dumps(configuration),
            model_versions_json=_versions_to_json(model_versions),
            retriever_versions_json=_versions_to_json(retriever_versions),
            graph_versions_json=_versions_to_json(graph_versions),
            reranker=reranker,
            ontology_version=ontology_version,
            dataset=dataset,
            timestamp=now.isoformat(),
            status=ExperimentStatus.RUNNING.value,
            metrics_json="{}",
            duration=None,
            random_seed=random_seed,
            git_commit=git_commit,
            error=None,
            started_at_epoch=now.timestamp(),
        )
        async with self._session_factory() as session:
            session.add(record)
            await session.commit()
        return _record_to_experiment(record)

    async def complete(self, experiment_id: str, metrics: dict[str, float]) -> Experiment:
        async with self._session_factory() as session:
            record = await session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise KeyError(f"No experiment with id {experiment_id!r}")
            record.status = ExperimentStatus.COMPLETED.value
            record.metrics_json = json.dumps(metrics)
            record.duration = dt.datetime.now(dt.timezone.utc).timestamp() - record.started_at_epoch
            await session.commit()
            return _record_to_experiment(record)

    async def fail(self, experiment_id: str, error: str) -> Experiment:
        async with self._session_factory() as session:
            record = await session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise KeyError(f"No experiment with id {experiment_id!r}")
            record.status = ExperimentStatus.FAILED.value
            record.error = error
            record.duration = dt.datetime.now(dt.timezone.utc).timestamp() - record.started_at_epoch
            await session.commit()
            return _record_to_experiment(record)

    async def get(self, experiment_id: str) -> Experiment | None:
        async with self._session_factory() as session:
            record = await session.get(ExperimentRecord, experiment_id)
            return _record_to_experiment(record) if record else None

    async def list(self) -> list[Experiment]:
        async with self._session_factory() as session:
            result = await session.execute(select(ExperimentRecord).order_by(ExperimentRecord.timestamp.desc()))
            return [_record_to_experiment(r) for r in result.scalars().all()]
