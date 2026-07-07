"""Prediction Engine (M4 Component 3): PredictionTask -> Feature Builder ->
Prediction Model -> Confidence -> Explanation.

Feature-generation and prediction latency are both recorded through
LearningRepository.record_model_performance (the same audit trail as model
accuracy), rather than a separate ad-hoc metrics channel — one place to
query "how is this model actually performing," covering both speed and
quality.
"""

import time

from app.features.extractor import FeatureExtractor
from app.prediction.models import Prediction, PredictionModel, PredictionTaskType
from app.repositories.learning_repository import LearningRepositoryProtocol

_FEATURE_LATENCY_METRIC = "feature_generation_latency_seconds"
_PREDICTION_LATENCY_METRIC = "prediction_latency_seconds"


class PredictionEngine:
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        models: dict[PredictionTaskType, PredictionModel],
        learning_repository: LearningRepositoryProtocol,
        model_name: str = "rule_based_v1",
    ):
        self._feature_extractor = feature_extractor
        self._models = models
        self._learning_repository = learning_repository
        self._model_name = model_name

    def register_model(self, task_type: PredictionTaskType, model: PredictionModel) -> None:
        """Swap in a model for a task type at runtime — how a freshly
        trained MLPredictionModel (M6) replaces the rule-based default
        without restarting the app or touching any other wiring."""
        self._models[task_type] = model

    async def predict(self, equipment_id: str, task_type: PredictionTaskType) -> Prediction:
        model = self._models.get(task_type)
        if model is None:
            raise ValueError(f"No prediction model registered for task type {task_type}")

        feature_start = time.perf_counter()
        features = await self._feature_extractor.extract(equipment_id)
        feature_latency = time.perf_counter() - feature_start

        predict_start = time.perf_counter()
        result = model.predict(features)
        predict_latency = time.perf_counter() - predict_start

        # An ML-backed model (M6) reports its own version via
        # result.model_version; rule-based models (M4) leave it unset and
        # fall back to the engine-level default label.
        model_name = result.model_version or self._model_name

        await self._learning_repository.record_model_performance(
            task_type.value, model_name, _FEATURE_LATENCY_METRIC, round(feature_latency, 4)
        )
        await self._learning_repository.record_model_performance(
            task_type.value, model_name, _PREDICTION_LATENCY_METRIC, round(predict_latency, 4)
        )

        prediction_id = await self._learning_repository.record_prediction(
            task_type=task_type.value,
            equipment_id=equipment_id,
            value=result.value,
            confidence=result.confidence,
            explanation=result.explanation,
            model_name=model_name,
        )

        return Prediction(
            id=prediction_id,
            task_type=task_type,
            equipment_id=equipment_id,
            value=result.value,
            confidence=result.confidence,
            explanation=result.explanation,
            model_name=model_name,
            features=features,
            uncertainty=result.uncertainty,
            model_version=result.model_version,
        )
