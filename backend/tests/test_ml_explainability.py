from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from app.ml.datasets.synthetic import generate_failure_prediction_dataset
from app.ml.explainability.explainer import Explainer


def _xy():
    df = generate_failure_prediction_dataset(200, seed=42)
    return df.drop(columns=["label"]), df["label"]


def test_feature_importance_uses_feature_importances_attribute():
    X, y = _xy()
    estimator = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    explainer = Explainer()

    results = explainer.feature_importance(estimator, list(X.columns))

    assert len(results) == len(X.columns)
    values = [r.importance for r in results]
    assert values == sorted(values, reverse=True)


def test_feature_importance_respects_top_n():
    X, y = _xy()
    estimator = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    explainer = Explainer()

    results = explainer.feature_importance(estimator, list(X.columns), top_n=3)

    assert len(results) == 3


def test_feature_importance_uses_coef_attribute_for_linear_models():
    X, y = _xy()
    estimator = LogisticRegression(max_iter=1000).fit(X, y)
    explainer = Explainer()

    results = explainer.feature_importance(estimator, list(X.columns))

    assert len(results) == len(X.columns)
    assert all(r.importance >= 0 for r in results)


def test_feature_importance_returns_empty_list_when_unsupported():
    class _NoImportanceEstimator:
        pass

    explainer = Explainer()
    assert explainer.feature_importance(_NoImportanceEstimator(), ["a", "b"]) == []


def test_permutation_importance_returns_one_result_per_column():
    X, y = _xy()
    estimator = RandomForestClassifier(n_estimators=30, random_state=42).fit(X, y)
    explainer = Explainer()

    results = explainer.permutation_importance(estimator, X, y, n_repeats=3)

    assert len(results) == len(X.columns)
    assert {r.feature_name for r in results} == set(X.columns)


def test_calibration_curve_returns_points_within_valid_range():
    X, y = _xy()
    estimator = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    explainer = Explainer()

    points = explainer.calibration_curve(estimator, X, y, n_bins=5)

    assert len(points) > 0
    for point in points:
        assert 0.0 <= point.predicted_probability <= 1.0
        assert 0.0 <= point.observed_frequency <= 1.0


def test_calibration_curve_returns_empty_without_predict_proba():
    from sklearn.ensemble import RandomForestRegressor

    X, y = _xy()
    estimator = RandomForestRegressor(n_estimators=10, random_state=42).fit(X, y.astype(float))
    explainer = Explainer()

    assert explainer.calibration_curve(estimator, X, y.astype(float)) == []


def test_decision_path_returns_readable_threshold_steps():
    X, y = _xy()
    estimator = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    explainer = Explainer()

    steps = explainer.decision_path(estimator, X.iloc[[0]])

    assert isinstance(steps, list)
    assert all(isinstance(step, str) for step in steps)


def test_decision_path_uses_first_estimator_for_forest_models():
    X, y = _xy()
    estimator = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42).fit(X, y)
    explainer = Explainer()

    steps = explainer.decision_path(estimator, X.iloc[[0]])

    assert isinstance(steps, list)


def test_decision_path_returns_empty_for_estimator_without_tree():
    class _NoTreeEstimator:
        pass

    X, _ = _xy()
    explainer = Explainer()

    assert explainer.decision_path(_NoTreeEstimator(), X.iloc[[0]]) == []
