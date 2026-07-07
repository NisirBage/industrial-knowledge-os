"""Explainability Layer (M6 Component 6).

feature_importance, permutation_importance, calibration_curve, and
decision_path are all built on scikit-learn's own inspection utilities —
real computations, zero new dependencies. SHAP (TreeExplainer/KernelExplainer
value attribution) is the documented next step: it would slot in as another
method on this same `Explainer` class returning the same
`list[FeatureImportanceResult]` shape, so nothing downstream (MLPredictionModel,
the ML Lab API) would need to change to consume it.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

from app.ml.training.models import SklearnEstimator


@dataclass(frozen=True)
class FeatureImportanceResult:
    feature_name: str
    importance: float


@dataclass(frozen=True)
class CalibrationPoint:
    predicted_probability: float
    observed_frequency: float


class Explainer:
    def feature_importance(
        self, estimator: SklearnEstimator, feature_names: list[str], top_n: int | None = None
    ) -> list[FeatureImportanceResult]:
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            importances = abs(coef).flatten() if hasattr(coef, "flatten") else [abs(c) for c in coef]
        else:
            return []

        pairs = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)
        if top_n is not None:
            pairs = pairs[:top_n]
        return [FeatureImportanceResult(name, round(float(value), 4)) for name, value in pairs]

    def permutation_importance(
        self, estimator: SklearnEstimator, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42
    ) -> list[FeatureImportanceResult]:
        result = permutation_importance(estimator, X, y, n_repeats=n_repeats, random_state=random_state)
        pairs = sorted(zip(X.columns, result.importances_mean), key=lambda pair: pair[1], reverse=True)
        return [FeatureImportanceResult(name, round(float(value), 4)) for name, value in pairs]

    def calibration_curve(self, estimator: SklearnEstimator, X: pd.DataFrame, y: pd.Series, n_bins: int = 10) -> list[CalibrationPoint]:
        if not hasattr(estimator, "predict_proba"):
            return []
        probabilities = estimator.predict_proba(X)[:, 1]
        observed, predicted = calibration_curve(y, probabilities, n_bins=n_bins, strategy="quantile")
        return [
            CalibrationPoint(predicted_probability=round(float(p), 4), observed_frequency=round(float(o), 4))
            for p, o in zip(predicted, observed, strict=True)
        ]

    def decision_path(self, estimator: SklearnEstimator, X_row: pd.DataFrame) -> list[str]:
        tree = estimator.estimators_[0] if hasattr(estimator, "estimators_") else estimator
        if not hasattr(tree, "tree_"):
            return []

        node_indicator = tree.decision_path(X_row)
        leaf_id = tree.apply(X_row)[0]
        feature_index = tree.tree_.feature
        threshold = tree.tree_.threshold
        visited_nodes = node_indicator.indices[node_indicator.indptr[0] : node_indicator.indptr[1]]

        steps = []
        for node_id in visited_nodes:
            if node_id == leaf_id:
                continue
            column_name = X_row.columns[feature_index[node_id]]
            direction = "<=" if X_row.iloc[0, feature_index[node_id]] <= threshold[node_id] else ">"
            steps.append(f"{column_name} {direction} {round(float(threshold[node_id]), 3)}")
        return steps
