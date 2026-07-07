"""Inference Engine (M6 Component 5): MLPredictionModel implements the exact
same `app.prediction.models.PredictionModel` protocol M4's
RuleBasedPredictionModel does — it's a drop-in replacement inserted into
PredictionEngine's `models: dict[PredictionTaskType, PredictionModel]` dict,
one task type at a time. Nothing about PredictionEngine changes; a model
that hasn't been trained yet for a given task simply keeps using the
rule-based fallback already registered there.

Uncertainty is a simple, honest proxy — distance from a confident (0 or 1)
classification probability, or 0.0 (a plain point estimate, no
uncertainty quantification) for regressors that don't expose one. A
real prediction interval (e.g. via quantile regression or an ensemble
spread) is the natural upgrade path, not implemented here.
"""

import pandas as pd

from app.features.models import EquipmentFeatureVector
from app.ml.explainability.explainer import Explainer
from app.ml.registry.models import ModelRecord
from app.ml.training.models import SklearnEstimator
from app.prediction.models import PredictionResult, PredictionTaskType

_TOP_FEATURES_IN_EXPLANATION = 3


def _feature_row(features: EquipmentFeatureVector, feature_schema: list[str]) -> pd.DataFrame:
    import dataclasses

    values = dataclasses.asdict(features)
    row = {name: values.get(name) if isinstance(values.get(name), (int, float)) else 0.0 for name in feature_schema}
    return pd.DataFrame([row])[feature_schema]


class MLPredictionModel:
    def __init__(
        self,
        task_type: PredictionTaskType,
        estimator: SklearnEstimator,
        model_record: ModelRecord,
        explainer: Explainer,
    ):
        self.task_type = task_type
        self._estimator = estimator
        self._model_record = model_record
        self._explainer = explainer

    def predict(self, features: EquipmentFeatureVector) -> PredictionResult:
        X = _feature_row(features, self._model_record.feature_schema)

        if hasattr(self._estimator, "predict_proba"):
            probabilities = self._estimator.predict_proba(X)[0]
            positive_index = 1 if len(probabilities) > 1 else 0
            value = float(probabilities[positive_index])
            uncertainty = round(1.0 - float(max(probabilities)), 4)
        else:
            value = float(self._estimator.predict(X)[0])
            uncertainty = 0.0

        confidence = round(1.0 - uncertainty, 4)
        top_features = self._explainer.feature_importance(self._estimator, self._model_record.feature_schema, top_n=_TOP_FEATURES_IN_EXPLANATION)
        feature_summary = ", ".join(f.feature_name for f in top_features) or "no feature importance available"

        model_version = f"{self._model_record.algorithm}:{self._model_record.model_id[:8]}"
        explanation = f"ML model ({model_version}) — top factors: {feature_summary}."

        return PredictionResult(
            value=round(value, 4),
            confidence=confidence,
            explanation=explanation,
            uncertainty=uncertainty,
            model_version=model_version,
        )
