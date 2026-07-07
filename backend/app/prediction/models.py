"""Prediction Engine data model (M4 Component 3).

PredictionModel is the swap point future ML (XGBoost, LightGBM, CatBoost,
PyTorch, scikit-learn) plugs into: anything implementing
`predict(EquipmentFeatureVector) -> PredictionResult` works, whether it's a
handful of if-statements (today) or a loaded model.predict() call wrapped in
a thin adapter (tomorrow) — PredictionEngine and every caller are unaffected
either way.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from app.features.models import EquipmentFeatureVector


class PredictionTaskType(str, Enum):
    REMAINING_USEFUL_LIFE = "remaining_useful_life"
    FAILURE_RISK = "failure_risk"
    INSPECTION_PRIORITY = "inspection_priority"
    MAINTENANCE_PRIORITY = "maintenance_priority"
    RECOMMENDATION_PRIORITY = "recommendation_priority"
    MISSING_DOCUMENTATION_RISK = "missing_documentation_risk"
    COMPLIANCE_RISK = "compliance_risk"


@dataclass(frozen=True)
class PredictionResult:
    value: float | None
    confidence: float
    explanation: str
    # Both optional and None-by-default so M4's rule-based models (which have
    # no statistical uncertainty estimate or model version to report) need
    # no changes — only M6's MLPredictionModel populates these.
    uncertainty: float | None = None
    model_version: str | None = None


class PredictionModel(Protocol):
    task_type: PredictionTaskType

    def predict(self, features: EquipmentFeatureVector) -> PredictionResult: ...


@dataclass(frozen=True)
class Prediction:
    id: str
    task_type: PredictionTaskType
    equipment_id: str
    value: float | None
    confidence: float
    explanation: str
    model_name: str
    features: EquipmentFeatureVector
    uncertainty: float | None = None
    model_version: str | None = None
