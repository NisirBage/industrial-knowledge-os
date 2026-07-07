"""Model Registry (M6 Component 4): every trained model is versioned,
serialized to disk via joblib, and recorded in SQLite — training_data_hash,
hyperparameters, metrics, and feature_schema together make a registered
model's training run reproducible and auditable, the same guarantee M5's
Experiment Manager gives at the experiment level (registry entries link to
an experiment_id when one exists).

ONNX/TorchScript serialization and MLflow-backed storage are documented
future formats — `serialization_format` is already a plain string field so
adding them later is a new value, not a schema change.
"""

import datetime as dt
import hashlib
import json
import uuid
from pathlib import Path
from typing import Protocol

import joblib
import pandas as pd
from sqlalchemy import String, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.ml.registry.models import ModelRecord
from app.ml.training.models import SklearnEstimator


class Base(DeclarativeBase):
    pass


class ModelRecordRow(Base):
    __tablename__ = "ml_models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String)
    algorithm: Mapped[str] = mapped_column(String)
    training_data_hash: Mapped[str] = mapped_column(String)
    hyperparameters_json: Mapped[str] = mapped_column(String)
    metrics_json: Mapped[str] = mapped_column(String)
    feature_schema_json: Mapped[str] = mapped_column(String)
    timestamp: Mapped[str] = mapped_column(String)
    author: Mapped[str] = mapped_column(String)
    experiment_id: Mapped[str | None] = mapped_column(String, nullable=True)
    serialization_path: Mapped[str] = mapped_column(String)
    serialization_format: Mapped[str] = mapped_column(String)


def hash_dataframe(df: pd.DataFrame) -> str:
    """A stable content hash of a training DataFrame — same data in, same
    hash out, regardless of process/run, so two ModelRecords can be
    compared to see if they were trained on identical data."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _row_to_record(row: ModelRecordRow) -> ModelRecord:
    return ModelRecord(
        model_id=row.model_id,
        task_type=row.task_type,
        algorithm=row.algorithm,
        training_data_hash=row.training_data_hash,
        hyperparameters=json.loads(row.hyperparameters_json),
        metrics=json.loads(row.metrics_json),
        feature_schema=json.loads(row.feature_schema_json),
        timestamp=row.timestamp,
        author=row.author,
        experiment_id=row.experiment_id,
        serialization_path=row.serialization_path,
        serialization_format=row.serialization_format,
    )


class ModelRegistryProtocol(Protocol):
    async def init_models(self) -> None: ...

    async def register(
        self,
        estimator: SklearnEstimator,
        task_type: str,
        algorithm: str,
        training_data_hash: str,
        hyperparameters: dict[str, str],
        metrics: dict[str, float],
        feature_schema: list[str],
        author: str,
        experiment_id: str | None,
    ) -> ModelRecord: ...

    async def load_estimator(self, model_id: str) -> SklearnEstimator: ...
    async def get(self, model_id: str) -> ModelRecord | None: ...
    async def get_latest(self, task_type: str) -> ModelRecord | None: ...
    async def list(self, task_type: str | None = None) -> list[ModelRecord]: ...


class SqlModelRegistry:
    def __init__(self, db_url: str, models_dir: Path):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)
        self._models_dir = models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def register(
        self,
        estimator: SklearnEstimator,
        task_type: str,
        algorithm: str,
        training_data_hash: str,
        hyperparameters: dict[str, str],
        metrics: dict[str, float],
        feature_schema: list[str],
        author: str = "system",
        experiment_id: str | None = None,
    ) -> ModelRecord:
        model_id = str(uuid.uuid4())
        serialization_path = str(self._models_dir / f"{model_id}.joblib")
        joblib.dump(estimator, serialization_path)

        row = ModelRecordRow(
            model_id=model_id,
            task_type=task_type,
            algorithm=algorithm,
            training_data_hash=training_data_hash,
            hyperparameters_json=json.dumps(hyperparameters),
            metrics_json=json.dumps(metrics),
            feature_schema_json=json.dumps(feature_schema),
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            author=author,
            experiment_id=experiment_id,
            serialization_path=serialization_path,
            serialization_format="joblib",
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
        return _row_to_record(row)

    async def load_estimator(self, model_id: str) -> SklearnEstimator:
        record = await self.get(model_id)
        if record is None:
            raise KeyError(f"No registered model with id {model_id!r}")
        return joblib.load(record.serialization_path)

    async def get(self, model_id: str) -> ModelRecord | None:
        async with self._session_factory() as session:
            row = await session.get(ModelRecordRow, model_id)
            return _row_to_record(row) if row else None

    async def get_latest(self, task_type: str) -> ModelRecord | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(ModelRecordRow).where(ModelRecordRow.task_type == task_type).order_by(ModelRecordRow.timestamp.desc()).limit(1)
            )
            row = result.scalar_one_or_none()
            return _row_to_record(row) if row else None

    async def list(self, task_type: str | None = None) -> list[ModelRecord]:
        async with self._session_factory() as session:
            query = select(ModelRecordRow).order_by(ModelRecordRow.timestamp.desc())
            if task_type is not None:
                query = query.where(ModelRecordRow.task_type == task_type)
            result = await session.execute(query)
            return [_row_to_record(row) for row in result.scalars().all()]
