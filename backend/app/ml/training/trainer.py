"""Model Training Framework (M6 Component 3): a generic trainer whose only
sklearn-specific piece is the default algorithm-per-problem-type table —
every algorithm is injected via `algorithm_factories`, so "no trainer
depends on a specific algorithm" holds even though only scikit-learn is
wired up today. An XGBoost/LightGBM/PyTorch trainer would implement the same
`Trainer` protocol and could be swapped in via the same constructor
argument without touching ModelRegistry, InferenceEngine, or anything else
downstream.

TIME_SERIES and GRAPH problem types have no default algorithm — scikit-learn
doesn't have a natural one-size-fits-all estimator for either, and faking
one would produce a "trained model" that's really just noise. `train()`
raises a clear error for them unless the caller supplies their own factory.
"""

import datetime as dt
from collections import defaultdict
from typing import Callable, Protocol

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from app.ml.datasets.models import MLProblemType
from app.ml.training.models import SklearnEstimator, TrainedModel


class Trainer(Protocol):
    def train(self, X: pd.DataFrame, y: pd.Series, problem_type: MLProblemType, task_type: str) -> TrainedModel: ...


_DEFAULT_ALGORITHM_FACTORIES: dict[MLProblemType, Callable[[], SklearnEstimator]] = {
    MLProblemType.CLASSIFICATION: lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    MLProblemType.REGRESSION: lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    MLProblemType.RANKING: lambda: RandomForestRegressor(n_estimators=200, random_state=42),
    MLProblemType.ANOMALY_DETECTION: lambda: IsolationForest(random_state=42),
}


class SklearnTrainer:
    def __init__(self, algorithm_factories: dict[MLProblemType, Callable[[], SklearnEstimator]] | None = None):
        self._algorithm_factories = {**_DEFAULT_ALGORITHM_FACTORIES, **(algorithm_factories or {})}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: MLProblemType,
        task_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> TrainedModel:
        factory = self._algorithm_factories.get(problem_type)
        if factory is None:
            raise ValueError(
                f"No algorithm registered for problem type {problem_type.value!r} — "
                f"supply one via SklearnTrainer(algorithm_factories={{...}})."
            )
        estimator = factory()

        X_clean = X.select_dtypes(include=["number"]).fillna(0.0)
        stratify = y if problem_type == MLProblemType.CLASSIFICATION else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        estimator.fit(X_train, y_train)
        metrics = self._evaluate(estimator, X_test, y_test, problem_type)

        return TrainedModel(
            estimator=estimator,
            task_type=task_type,
            algorithm=type(estimator).__name__,
            feature_names=list(X_clean.columns),
            metrics=metrics,
            trained_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            X_test=X_test,
            y_test=y_test,
            hyperparameters={key: str(value) for key, value in estimator.get_params().items()},
        )

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: MLProblemType,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """K-fold cross-validation, distinct from `train()`'s single
        held-out split: fits a fresh estimator per fold and averages the same
        metrics `_evaluate` computes, giving a variance-aware complement to
        the single-split point estimate (see the v1.0.1 audit finding on
        cross-validation being implemented in TrainingDatasetBuilder but
        never actually exercised in the live training path)."""
        factory = self._algorithm_factories.get(problem_type)
        if factory is None:
            raise ValueError(
                f"No algorithm registered for problem type {problem_type.value!r} — "
                f"supply one via SklearnTrainer(algorithm_factories={{...}})."
            )

        X_clean = X.select_dtypes(include=["number"]).fillna(0.0)

        if problem_type == MLProblemType.CLASSIFICATION:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = splitter.split(X_clean, y)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = splitter.split(X_clean)

        fold_metrics: list[dict[str, float]] = []
        for train_idx, test_idx in splits:
            estimator = factory()
            estimator.fit(X_clean.iloc[train_idx], y.iloc[train_idx])
            fold_metrics.append(self._evaluate(estimator, X_clean.iloc[test_idx], y.iloc[test_idx], problem_type))

        return self._average_fold_metrics(fold_metrics)

    @staticmethod
    def _average_fold_metrics(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
        if not fold_metrics:
            return {}
        totals: dict[str, list[float]] = defaultdict(list)
        for metrics in fold_metrics:
            for key, value in metrics.items():
                totals[key].append(value)
        return {key: round(sum(values) / len(values), 4) for key, values in totals.items()}

    @staticmethod
    def _evaluate(estimator: SklearnEstimator, X_test: pd.DataFrame, y_test: pd.Series, problem_type: MLProblemType) -> dict[str, float]:
        if problem_type == MLProblemType.CLASSIFICATION:
            y_pred = estimator.predict(X_test)
            metrics = {
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            }
            if hasattr(estimator, "predict_proba") and y_test.nunique() == 2:
                proba = estimator.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, proba)), 4)
                metrics["pr_auc"] = round(float(average_precision_score(y_test, proba)), 4)
                metrics["brier_score"] = round(float(brier_score_loss(y_test, proba)), 4)
            return metrics

        if problem_type == MLProblemType.REGRESSION:
            y_pred = estimator.predict(X_test)
            return {
                "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
                "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            }

        return {}
