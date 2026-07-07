from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


class SklearnEstimator(Protocol):
    """The minimal duck-typed surface every sklearn-compatible estimator
    exposes — used instead of `Any` so TrainedModel.estimator still has a
    real (if loose) type."""

    def fit(self, X, y=None): ...
    def predict(self, X): ...


@dataclass
class TrainedModel:
    estimator: SklearnEstimator
    task_type: str
    algorithm: str
    feature_names: list[str]
    metrics: dict[str, float]
    trained_at: str
    # The trainer's own held-out split — callers needing further diagnostics
    # (confusion matrix, ROC, calibration) must evaluate on this, not the
    # full dataset, or those diagnostics silently include training rows.
    X_test: pd.DataFrame
    y_test: pd.Series
    hyperparameters: dict[str, str] = field(default_factory=dict)
