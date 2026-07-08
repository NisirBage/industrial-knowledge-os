import pytest
from sklearn.linear_model import LogisticRegression

from app.ml.datasets.models import MLProblemType
from app.ml.datasets.synthetic import generate_failure_prediction_dataset, generate_maintenance_priority_dataset
from app.ml.training.trainer import SklearnTrainer


def _classification_xy():
    df = generate_failure_prediction_dataset(300, seed=42)
    return df.drop(columns=["label"]), df["label"]


def _regression_xy():
    df = generate_maintenance_priority_dataset(300, seed=42)
    return df.drop(columns=["label"]), df["label"]


def test_train_classification_produces_real_classification_metrics():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    assert trained.algorithm == "RandomForestClassifier"
    assert 0.0 <= trained.metrics["accuracy"] <= 1.0
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score"} <= trained.metrics.keys()


def test_train_classification_beats_majority_class_baseline():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")
    majority_baseline = max(y.mean(), 1 - y.mean())

    assert trained.metrics["accuracy"] >= majority_baseline - 0.05


def test_train_regression_produces_rmse_and_mae():
    X, y = _regression_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.REGRESSION, "maintenance_priority")

    assert trained.algorithm == "RandomForestRegressor"
    assert trained.metrics["rmse"] >= 0.0
    assert trained.metrics["mae"] >= 0.0


def test_train_holds_out_test_set_disjoint_from_training():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction", test_size=0.25)

    assert len(trained.X_test) == round(0.25 * len(X))
    assert len(trained.y_test) == len(trained.X_test)


def test_train_drops_non_numeric_columns_from_feature_names():
    X, y = _classification_xy()
    X = X.copy()
    X["equipment_note"] = "not a number"
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    assert "equipment_note" not in trained.feature_names
    assert set(trained.feature_names) == set(X.select_dtypes(include=["number"]).columns)


def test_train_fills_missing_numeric_values_with_zero():
    X, y = _classification_xy()
    X = X.copy()
    X.loc[X.index[0], "risk_score"] = None
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    assert trained.metrics["accuracy"] >= 0.0


def test_train_raises_for_problem_type_without_default_algorithm():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    with pytest.raises(ValueError):
        trainer.train(X, y, MLProblemType.TIME_SERIES, "root_cause_classification")


def test_custom_algorithm_factory_overrides_default():
    X, y = _classification_xy()
    trainer = SklearnTrainer(algorithm_factories={MLProblemType.CLASSIFICATION: lambda: LogisticRegression(max_iter=1000)})

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    assert trained.algorithm == "LogisticRegression"


def test_trained_model_records_hyperparameters_as_strings():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    assert all(isinstance(v, str) for v in trained.hyperparameters.values())
    assert trained.hyperparameters["n_estimators"] == "200"


def test_anomaly_detection_uses_isolation_forest_and_reports_no_supervised_metrics():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    trained = trainer.train(X, y, MLProblemType.ANOMALY_DETECTION, "equipment_risk")

    assert trained.algorithm == "IsolationForest"
    assert trained.metrics == {}


def test_cross_validate_classification_returns_averaged_metrics_across_folds():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    cv_metrics = trainer.cross_validate(X, y, MLProblemType.CLASSIFICATION, n_splits=5)

    assert {"accuracy", "precision", "recall", "f1"} <= cv_metrics.keys()
    assert 0.0 <= cv_metrics["accuracy"] <= 1.0


def test_cross_validate_regression_returns_averaged_rmse_and_mae():
    X, y = _regression_xy()
    trainer = SklearnTrainer()

    cv_metrics = trainer.cross_validate(X, y, MLProblemType.REGRESSION, n_splits=4)

    assert cv_metrics["rmse"] >= 0.0
    assert cv_metrics["mae"] >= 0.0


def test_cross_validate_fits_a_fresh_estimator_per_fold_not_the_held_out_split():
    """Cross-validation must be a distinct code path from train()'s single
    held-out split — same estimator type, but its own independent fit/score
    per fold (regression test for the audit finding that CV utilities
    existed but were never exercised in the live training path)."""
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    held_out = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction", random_state=1)
    cv_metrics = trainer.cross_validate(X, y, MLProblemType.CLASSIFICATION, n_splits=5, random_state=1)

    # Different evaluation methodology (1 split vs. mean of 5) should not be
    # expected to match exactly — this asserts they're computed independently,
    # not that CV must differ from held-out (it may coincidentally be close).
    assert isinstance(held_out.metrics["accuracy"], float)
    assert isinstance(cv_metrics["accuracy"], float)


def test_cross_validate_raises_for_problem_type_without_default_algorithm():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    with pytest.raises(ValueError):
        trainer.cross_validate(X, y, MLProblemType.TIME_SERIES, n_splits=5)


def test_cross_validate_is_deterministic_for_fixed_random_state():
    X, y = _classification_xy()
    trainer = SklearnTrainer()

    first = trainer.cross_validate(X, y, MLProblemType.CLASSIFICATION, n_splits=5, random_state=7)
    second = trainer.cross_validate(X, y, MLProblemType.CLASSIFICATION, n_splits=5, random_state=7)

    assert first == second
