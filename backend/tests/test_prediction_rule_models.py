from app.features.models import EquipmentFeatureVector
from app.prediction.models import PredictionTaskType
from app.prediction.rule_models import build_rule_based_models


def _features(**overrides) -> EquipmentFeatureVector:
    base = dict(
        equipment_id="P-101",
        days_since_last_failure=None,
        maintenance_frequency=1.0,
        inspection_frequency=1.0,
        failure_count=1,
        incident_count=1,
        average_repair_time=None,
        recommendation_acceptance_rate=None,
        contradiction_rate=0.0,
        risk_score=0.2,
        equipment_age=None,
        manufacturer=None,
        location=None,
        criticality="low",
        ontology_depth=2,
        graph_degree=5,
        retrieval_frequency=0,
    )
    base.update(overrides)
    return EquipmentFeatureVector(**base)


def test_failure_risk_uses_risk_score_as_value():
    models = build_rule_based_models()
    result = models[PredictionTaskType.FAILURE_RISK].predict(_features(risk_score=0.42))
    assert result.value == 0.42


def test_remaining_useful_life_returns_none_without_equipment_age():
    models = build_rule_based_models()
    result = models[PredictionTaskType.REMAINING_USEFUL_LIFE].predict(_features(equipment_age=None))
    assert result.value is None
    assert result.confidence == 0.0


def test_remaining_useful_life_estimates_when_age_known():
    models = build_rule_based_models()
    result = models[PredictionTaskType.REMAINING_USEFUL_LIFE].predict(_features(equipment_age=5, risk_score=0.2))
    assert result.value is not None
    assert result.value >= 0.0


def test_inspection_priority_is_urgent_with_incidents_and_no_inspections():
    models = build_rule_based_models()
    result = models[PredictionTaskType.INSPECTION_PRIORITY].predict(_features(incident_count=2, inspection_frequency=0))
    assert result.value == 1.0


def test_maintenance_priority_bounded_to_one():
    models = build_rule_based_models()
    result = models[PredictionTaskType.MAINTENANCE_PRIORITY].predict(_features(risk_score=0.95, maintenance_frequency=0))
    assert result.value == 1.0


def test_recommendation_priority_confidence_reflects_acceptance_history():
    models = build_rule_based_models()
    with_history = models[PredictionTaskType.RECOMMENDATION_PRIORITY].predict(_features(recommendation_acceptance_rate=0.9))
    without_history = models[PredictionTaskType.RECOMMENDATION_PRIORITY].predict(_features(recommendation_acceptance_rate=None))
    assert with_history.confidence > without_history.confidence


def test_missing_documentation_risk_high_when_graph_degree_low():
    models = build_rule_based_models()
    result = models[PredictionTaskType.MISSING_DOCUMENTATION_RISK].predict(_features(graph_degree=1))
    assert result.value == 0.8


def test_missing_documentation_risk_low_when_graph_degree_high():
    models = build_rule_based_models()
    result = models[PredictionTaskType.MISSING_DOCUMENTATION_RISK].predict(_features(graph_degree=10))
    assert result.value == 0.0


def test_compliance_risk_increases_with_contradiction_rate():
    models = build_rule_based_models()
    low = models[PredictionTaskType.COMPLIANCE_RISK].predict(_features(contradiction_rate=0.1, graph_degree=10))
    high = models[PredictionTaskType.COMPLIANCE_RISK].predict(_features(contradiction_rate=0.9, graph_degree=10))
    assert high.value > low.value


def test_build_rule_based_models_covers_every_task_type():
    models = build_rule_based_models()
    assert set(models.keys()) == set(PredictionTaskType)
