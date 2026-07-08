from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from app.features.models import EquipmentFeatureVector
from app.ml.datasets.synthetic import generate_failure_prediction_dataset, generate_maintenance_priority_dataset
from app.ml.explainability.explainer import Explainer
from app.ml.inference.model import MLPredictionModel
from app.ml.registry.models import ModelRecord
from app.prediction.models import PredictionTaskType


def _feature_vector(**overrides) -> EquipmentFeatureVector:
    base = dict(
        equipment_id="P-101",
        days_since_last_failure=10,
        maintenance_frequency=2.0,
        inspection_frequency=1.0,
        failure_count=2,
        incident_count=1,
        average_repair_time=None,
        recommendation_acceptance_rate=None,
        contradiction_rate=0.2,
        risk_score=0.6,
        equipment_age=None,
        manufacturer=None,
        location=None,
        criticality="high",
        ontology_depth=1,
        graph_degree=5,
        retrieval_frequency=2,
    )
    base.update(overrides)
    return EquipmentFeatureVector(**base)


def _record(feature_schema: list[str], algorithm: str) -> ModelRecord:
    return ModelRecord(
        model_id="12345678-abcd-ef00-0000-000000000000",
        task_type="failure_prediction",
        algorithm=algorithm,
        training_data_hash="hash",
        hyperparameters={},
        metrics={},
        feature_schema=feature_schema,
        timestamp="2026-01-01T00:00:00+00:00",
        author="test",
        experiment_id=None,
        serialization_path="unused.joblib",
        serialization_format="joblib",
    )


def _trained_classifier():
    df = generate_failure_prediction_dataset(200, seed=42)
    X, y = df.drop(columns=["label"]), df["label"]
    estimator = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
    return estimator, list(X.columns)


def _trained_regressor():
    df = generate_maintenance_priority_dataset(200, seed=42)
    X, y = df.drop(columns=["label"]), df["label"]
    estimator = RandomForestRegressor(n_estimators=50, random_state=42).fit(X, y)
    return estimator, list(X.columns)


def test_predict_with_classifier_returns_probability_and_uncertainty():
    estimator, feature_names = _trained_classifier()
    model = MLPredictionModel(PredictionTaskType.FAILURE_RISK, estimator, _record(feature_names, "RandomForestClassifier"), Explainer())

    result = model.predict(_feature_vector())

    assert 0.0 <= result.value <= 1.0
    assert 0.0 <= result.uncertainty <= 1.0
    assert result.confidence == round(1.0 - result.uncertainty, 4)


def test_predict_with_regressor_has_zero_uncertainty_and_full_confidence():
    estimator, feature_names = _trained_regressor()
    model = MLPredictionModel(PredictionTaskType.MAINTENANCE_PRIORITY, estimator, _record(feature_names, "RandomForestRegressor"), Explainer())

    result = model.predict(_feature_vector())

    assert result.uncertainty == 0.0
    assert result.confidence == 1.0


def test_predict_explanation_includes_model_version_and_top_features():
    estimator, feature_names = _trained_classifier()
    record = _record(feature_names, "RandomForestClassifier")
    model = MLPredictionModel(PredictionTaskType.FAILURE_RISK, estimator, record, Explainer())

    result = model.predict(_feature_vector())

    expected_version = f"RandomForestClassifier:{record.model_id[:8]}"
    assert result.model_version == expected_version
    assert expected_version in result.explanation
    assert "top factors" in result.explanation


def test_predict_handles_none_valued_features_by_defaulting_to_zero():
    estimator, feature_names = _trained_classifier()
    model = MLPredictionModel(PredictionTaskType.FAILURE_RISK, estimator, _record(feature_names, "RandomForestClassifier"), Explainer())

    result = model.predict(_feature_vector(average_repair_time=None, equipment_age=None))

    assert result.value is not None


def test_predict_ignores_non_numeric_fields_not_in_feature_schema():
    estimator, feature_names = _trained_classifier()
    assert "criticality" not in feature_names
    model = MLPredictionModel(PredictionTaskType.FAILURE_RISK, estimator, _record(feature_names, "RandomForestClassifier"), Explainer())

    result_high = model.predict(_feature_vector(criticality="high"))
    result_low = model.predict(_feature_vector(criticality="low"))

    assert result_high.value == result_low.value
