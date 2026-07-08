"""Learning Repository (M4 Component 4): the durable, auditable record of
every prediction made, every outcome observed, every piece of human
feedback, and every model performance measurement.

This is the only place learning-related persistence happens — the
Prediction Engine and Feedback Engine never touch SQLAlchemy directly, only
this repository's typed methods (mirrors the pattern already established by
MetadataRepository/GraphRepository: business logic never sees the storage
layer's query language).

Deliberately synchronous with nothing else: this subsystem is independent of
the (synchronous) chat orchestrator this milestone — see module docstring in
app/prediction/engine.py for why.
"""

import datetime as dt
import uuid
from typing import Protocol

from sqlalchemy import Float, String, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String)
    equipment_id: Mapped[str] = mapped_column(String)
    value: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    explanation: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class OutcomeRecord(Base):
    __tablename__ = "outcomes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String)
    actual_value: Mapped[float] = mapped_column(Float)
    notes: Mapped[str] = mapped_column(String, default="")
    recorded_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String)
    target_id: Mapped[str] = mapped_column(String)
    # Recommendation ids (M3) aren't equipment-prefixed, so per-equipment
    # acceptance-rate queries need this as its own column, supplied by the
    # caller (FeedbackEngine), rather than parsed out of target_id.
    equipment_id: Mapped[str | None] = mapped_column(String, nullable=True)
    reason: Mapped[str | None] = mapped_column(String, nullable=True)
    corrected_value: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class ModelPerformanceRecord(Base):
    __tablename__ = "model_performance"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    task_type: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    metric_name: Mapped[str] = mapped_column(String)
    metric_value: Mapped[float] = mapped_column(Float)
    evaluated_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class InteractionRecord(Base):
    __tablename__ = "interactions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    equipment_id: Mapped[str] = mapped_column(String)
    query: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class ContradictionRecord(Base):
    __tablename__ = "learning_contradictions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    equipment_id: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    created_at: Mapped[dt.datetime] = mapped_column(default=dt.datetime.utcnow)


class LearningRepositoryProtocol(Protocol):
    async def init_models(self) -> None: ...
    async def record_prediction(
        self, task_type: str, equipment_id: str, value: float | None, confidence: float, explanation: str, model_name: str
    ) -> str: ...
    async def record_outcome(self, prediction_id: str, actual_value: float, notes: str = "") -> str: ...
    async def record_feedback(
        self, kind: str, target_id: str, reason: str | None, corrected_value: str | None, equipment_id: str | None = None
    ) -> str: ...
    async def record_model_performance(self, task_type: str, model_name: str, metric_name: str, metric_value: float) -> str: ...
    async def record_interaction(self, equipment_id: str, query: str) -> str: ...
    async def record_contradiction(self, equipment_id: str, description: str, severity: str) -> str: ...
    async def recommendation_acceptance_rate(self, equipment_id: str) -> float | None: ...
    async def contradiction_rate(self, equipment_id: str) -> float: ...
    async def retrieval_frequency(self, equipment_id: str) -> int: ...
    async def feedback_volume(self) -> int: ...
    async def average_prediction_confidence(self) -> float | None: ...
    async def latest_prediction_confidence(self, equipment_id: str) -> float | None: ...
    async def average_model_accuracy(self, metric_name: str = "accuracy") -> float | None: ...
    async def confidence_calibration_buckets(self, threshold: float = 0.2) -> list[dict]: ...


class SqlLearningRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def record_prediction(
        self, task_type: str, equipment_id: str, value: float | None, confidence: float, explanation: str, model_name: str
    ) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(
                PredictionRecord(
                    id=record_id,
                    task_type=task_type,
                    equipment_id=equipment_id,
                    value=value,
                    confidence=confidence,
                    explanation=explanation,
                    model_name=model_name,
                )
            )
            await session.commit()
        return record_id

    async def record_outcome(self, prediction_id: str, actual_value: float, notes: str = "") -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(OutcomeRecord(id=record_id, prediction_id=prediction_id, actual_value=actual_value, notes=notes))
            await session.commit()
        return record_id

    async def record_feedback(
        self, kind: str, target_id: str, reason: str | None, corrected_value: str | None, equipment_id: str | None = None
    ) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(
                FeedbackRecord(
                    id=record_id,
                    kind=kind,
                    target_id=target_id,
                    equipment_id=equipment_id,
                    reason=reason,
                    corrected_value=corrected_value,
                )
            )
            await session.commit()
        return record_id

    async def record_model_performance(self, task_type: str, model_name: str, metric_name: str, metric_value: float) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(
                ModelPerformanceRecord(
                    id=record_id, task_type=task_type, model_name=model_name, metric_name=metric_name, metric_value=metric_value
                )
            )
            await session.commit()
        return record_id

    async def record_interaction(self, equipment_id: str, query: str) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(InteractionRecord(id=record_id, equipment_id=equipment_id, query=query))
            await session.commit()
        return record_id

    async def record_contradiction(self, equipment_id: str, description: str, severity: str) -> str:
        record_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(ContradictionRecord(id=record_id, equipment_id=equipment_id, description=description, severity=severity))
            await session.commit()
        return record_id

    async def recommendation_acceptance_rate(self, equipment_id: str) -> float | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(FeedbackRecord.kind).where(FeedbackRecord.equipment_id == equipment_id)
            )
            kinds = [row[0] for row in result.all()]
        decisive = [k for k in kinds if k in ("recommendation_accepted", "recommendation_rejected")]
        if not decisive:
            return None
        accepted = sum(1 for k in decisive if k == "recommendation_accepted")
        return round(accepted / len(decisive), 4)

    async def contradiction_rate(self, equipment_id: str) -> float:
        async with self._session_factory() as session:
            contradiction_count = await session.scalar(
                select(func.count()).select_from(ContradictionRecord).where(ContradictionRecord.equipment_id == equipment_id)
            )
            interaction_count = await session.scalar(
                select(func.count()).select_from(InteractionRecord).where(InteractionRecord.equipment_id == equipment_id)
            )
        total = max(interaction_count or 0, 1)
        return round((contradiction_count or 0) / total, 4)

    async def retrieval_frequency(self, equipment_id: str) -> int:
        async with self._session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(InteractionRecord).where(InteractionRecord.equipment_id == equipment_id)
            )
        return count or 0

    async def feedback_volume(self) -> int:
        async with self._session_factory() as session:
            count = await session.scalar(select(func.count()).select_from(FeedbackRecord))
        return count or 0

    async def average_prediction_confidence(self) -> float | None:
        async with self._session_factory() as session:
            avg = await session.scalar(select(func.avg(PredictionRecord.confidence)))
        return round(avg, 4) if avg is not None else None

    async def latest_prediction_confidence(self, equipment_id: str) -> float | None:
        """Added for the Graph Feature Store (M9) — every prior consumer only
        needed the platform-wide average; per-node features need the most
        recent prediction confidence for *this* equipment specifically."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(PredictionRecord.confidence)
                .where(PredictionRecord.equipment_id == equipment_id)
                .order_by(PredictionRecord.created_at.desc())
                .limit(1)
            )
            row = result.first()
        return round(row[0], 4) if row else None

    async def average_model_accuracy(self, metric_name: str = "accuracy") -> float | None:
        async with self._session_factory() as session:
            avg = await session.scalar(
                select(func.avg(ModelPerformanceRecord.metric_value)).where(ModelPerformanceRecord.metric_name == metric_name)
            )
        return round(avg, 4) if avg is not None else None

    async def confidence_calibration_buckets(self, threshold: float = 0.2) -> list[dict]:
        """Buckets predictions by confidence decile and reports, for
        predictions with a recorded outcome, the fraction within `threshold`
        of the actual value ("correct" for calibration purposes — these are
        continuous risk scores, not classifications, so exact match isn't
        the right bar). Buckets with no outcomes yet report actual_accuracy
        as None rather than a misleading 0.0.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(PredictionRecord.confidence, OutcomeRecord.actual_value, PredictionRecord.value).join(
                    OutcomeRecord, OutcomeRecord.prediction_id == PredictionRecord.id, isouter=True
                )
            )
            rows = result.all()

        buckets: dict[str, list[tuple[float, bool | None]]] = {}
        for confidence, actual_value, predicted_value in rows:
            bucket_key = f"{int(confidence * 10) * 10}-{int(confidence * 10) * 10 + 10}%"
            is_correct = None
            if actual_value is not None and predicted_value is not None:
                is_correct = abs(predicted_value - actual_value) <= threshold
            buckets.setdefault(bucket_key, []).append((confidence, is_correct))

        report = []
        for bucket_key, entries in sorted(buckets.items()):
            confidences = [c for c, _ in entries]
            outcomes = [correct for _, correct in entries if correct is not None]
            report.append(
                {
                    "confidence_bucket": bucket_key,
                    "mean_confidence": round(sum(confidences) / len(confidences), 4),
                    "actual_accuracy": round(sum(outcomes) / len(outcomes), 4) if outcomes else None,
                    "sample_count": len(entries),
                }
            )
        return report
