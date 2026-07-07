import pytest

from app.features.models import EquipmentFeatureVector
from app.prediction.engine import PredictionEngine
from app.prediction.models import PredictionResult, PredictionTaskType
from app.prediction.rule_models import build_rule_based_models


class _FakeFeatureExtractor:
    async def extract(self, equipment_id: str) -> EquipmentFeatureVector:
        return EquipmentFeatureVector(
            equipment_id=equipment_id,
            days_since_last_failure=None,
            maintenance_frequency=1.0,
            inspection_frequency=1.0,
            failure_count=2,
            incident_count=2,
            average_repair_time=None,
            recommendation_acceptance_rate=None,
            contradiction_rate=0.1,
            risk_score=0.5,
            equipment_age=None,
            manufacturer=None,
            location=None,
            criticality="medium",
            ontology_depth=2,
            graph_degree=4,
            retrieval_frequency=0,
        )


class _RecordingLearningRepository:
    def __init__(self):
        self.predictions: list[dict] = []
        self.performance: list[tuple] = []

    async def record_prediction(self, task_type, equipment_id, value, confidence, explanation, model_name) -> str:
        self.predictions.append(
            {"task_type": task_type, "equipment_id": equipment_id, "value": value, "confidence": confidence, "model_name": model_name}
        )
        return f"pred-{len(self.predictions)}"

    async def record_model_performance(self, task_type, model_name, metric_name, metric_value) -> str:
        self.performance.append((task_type, model_name, metric_name, metric_value))
        return f"perf-{len(self.performance)}"


async def test_predict_returns_prediction_with_features_attached():
    learning_repo = _RecordingLearningRepository()
    engine = PredictionEngine(_FakeFeatureExtractor(), build_rule_based_models(), learning_repo)

    prediction = await engine.predict("P-101", PredictionTaskType.FAILURE_RISK)

    assert prediction.equipment_id == "P-101"
    assert prediction.task_type == PredictionTaskType.FAILURE_RISK
    assert prediction.features.risk_score == 0.5
    assert prediction.value == 0.5


async def test_predict_records_prediction_in_learning_repository():
    learning_repo = _RecordingLearningRepository()
    engine = PredictionEngine(_FakeFeatureExtractor(), build_rule_based_models(), learning_repo)

    await engine.predict("P-101", PredictionTaskType.FAILURE_RISK)

    assert len(learning_repo.predictions) == 1
    assert learning_repo.predictions[0]["equipment_id"] == "P-101"


async def test_predict_records_both_latency_metrics():
    learning_repo = _RecordingLearningRepository()
    engine = PredictionEngine(_FakeFeatureExtractor(), build_rule_based_models(), learning_repo)

    await engine.predict("P-101", PredictionTaskType.FAILURE_RISK)

    metric_names = {p[2] for p in learning_repo.performance}
    assert metric_names == {"feature_generation_latency_seconds", "prediction_latency_seconds"}


async def test_predict_raises_for_unknown_task_type():
    learning_repo = _RecordingLearningRepository()
    engine = PredictionEngine(_FakeFeatureExtractor(), {}, learning_repo)

    with pytest.raises(ValueError):
        await engine.predict("P-101", PredictionTaskType.FAILURE_RISK)


async def test_custom_model_can_replace_rule_based_model_without_interface_change():
    class _AlwaysHighRiskModel:
        task_type = PredictionTaskType.FAILURE_RISK

        def predict(self, features: EquipmentFeatureVector) -> PredictionResult:
            return PredictionResult(value=0.99, confidence=0.99, explanation="stub model for testing swap-in")

    learning_repo = _RecordingLearningRepository()
    engine = PredictionEngine(_FakeFeatureExtractor(), {PredictionTaskType.FAILURE_RISK: _AlwaysHighRiskModel()}, learning_repo)

    prediction = await engine.predict("P-101", PredictionTaskType.FAILURE_RISK)

    assert prediction.value == 0.99
    assert prediction.confidence == 0.99
