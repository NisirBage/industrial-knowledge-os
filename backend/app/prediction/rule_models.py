"""Rule-based baseline PredictionModel — one deterministic scorer per task
type. These are intentionally simple and explainable; they exist so the
Prediction Engine has something real to call before any statistical model
is trained. Each scorer is a pure function over EquipmentFeatureVector,
which is exactly the seam a trained model's `.predict()` call would occupy
later (see models.py's PredictionModel protocol).

Honesty over completeness: REMAINING_USEFUL_LIFE returns value=None because
its inputs (equipment_age) aren't captured by ingestion yet — no fabricated
number is better than a confident-looking wrong one.
"""

from typing import Callable

from app.features.models import EquipmentFeatureVector
from app.prediction.models import PredictionResult, PredictionTaskType

_LOW_GRAPH_DEGREE_THRESHOLD = 3


def _failure_risk(features: EquipmentFeatureVector) -> PredictionResult:
    history_signals = features.failure_count + features.incident_count
    confidence = round(min(0.3 + 0.1 * history_signals, 0.9), 2)
    return PredictionResult(
        value=features.risk_score,
        confidence=confidence,
        explanation=(
            f"Derived from {features.failure_count} failure(s) and {features.incident_count} incident(s), "
            f"with a contradiction rate of {features.contradiction_rate}."
        ),
    )


def _remaining_useful_life(features: EquipmentFeatureVector) -> PredictionResult:
    if features.equipment_age is None:
        return PredictionResult(
            value=None,
            confidence=0.0,
            explanation="Cannot estimate remaining useful life without equipment age/install-date data, which is not yet captured at ingestion.",
        )
    # Placeholder linear decay once age data exists — first real extension point for a survival-analysis model.
    estimated_years_remaining = max(0.0, 20.0 - features.equipment_age - 5.0 * features.risk_score)
    return PredictionResult(
        value=estimated_years_remaining,
        confidence=0.4,
        explanation=f"Linear estimate from equipment_age={features.equipment_age} and risk_score={features.risk_score}.",
    )


def _inspection_priority(features: EquipmentFeatureVector) -> PredictionResult:
    if features.incident_count > 0 and features.inspection_frequency == 0:
        return PredictionResult(
            value=1.0,
            confidence=0.6,
            explanation="Incidents on record but no inspections found — highest priority.",
        )
    value = min(1.0, (features.failure_count + features.incident_count) / (features.inspection_frequency + 1))
    return PredictionResult(
        value=round(value, 4),
        confidence=0.5,
        explanation=f"Ratio of {features.failure_count + features.incident_count} failure/incident event(s) to {features.inspection_frequency} inspection(s).",
    )


def _maintenance_priority(features: EquipmentFeatureVector) -> PredictionResult:
    value = min(1.0, features.risk_score + (0.1 if features.maintenance_frequency == 0 else 0.0))
    return PredictionResult(
        value=round(value, 4),
        confidence=0.5,
        explanation=f"Based on risk_score={features.risk_score} and maintenance_frequency={features.maintenance_frequency}.",
    )


def _recommendation_priority(features: EquipmentFeatureVector) -> PredictionResult:
    if features.recommendation_acceptance_rate is None:
        confidence = 0.5
        acceptance_note = "no recommendation history yet"
    else:
        confidence = round(0.4 + 0.4 * features.recommendation_acceptance_rate, 2)
        acceptance_note = f"historical acceptance rate {features.recommendation_acceptance_rate}"
    return PredictionResult(
        value=features.risk_score,
        confidence=confidence,
        explanation=f"Priority tracks risk_score={features.risk_score}; {acceptance_note}.",
    )


def _missing_documentation_risk(features: EquipmentFeatureVector) -> PredictionResult:
    if features.graph_degree < _LOW_GRAPH_DEGREE_THRESHOLD:
        value = 0.8
    else:
        value = round(max(0.0, 1 - features.graph_degree / 10), 4)
    return PredictionResult(
        value=value,
        confidence=0.4,
        explanation=f"Approximated from graph connectivity (graph_degree={features.graph_degree}); a weak proxy pending a dedicated documentation-completeness check.",
    )


def _compliance_risk(features: EquipmentFeatureVector) -> PredictionResult:
    value = min(1.0, features.contradiction_rate + (0.3 if features.graph_degree < _LOW_GRAPH_DEGREE_THRESHOLD else 0.0))
    return PredictionResult(
        value=round(value, 4),
        confidence=0.4,
        explanation=f"Approximated from contradiction_rate={features.contradiction_rate} and graph connectivity; a dedicated standards-linkage check is a natural extension.",
    )


_SCORERS: dict[PredictionTaskType, Callable[[EquipmentFeatureVector], PredictionResult]] = {
    PredictionTaskType.FAILURE_RISK: _failure_risk,
    PredictionTaskType.REMAINING_USEFUL_LIFE: _remaining_useful_life,
    PredictionTaskType.INSPECTION_PRIORITY: _inspection_priority,
    PredictionTaskType.MAINTENANCE_PRIORITY: _maintenance_priority,
    PredictionTaskType.RECOMMENDATION_PRIORITY: _recommendation_priority,
    PredictionTaskType.MISSING_DOCUMENTATION_RISK: _missing_documentation_risk,
    PredictionTaskType.COMPLIANCE_RISK: _compliance_risk,
}


class RuleBasedPredictionModel:
    def __init__(self, task_type: PredictionTaskType):
        self.task_type = task_type
        self._scorer = _SCORERS[task_type]

    def predict(self, features: EquipmentFeatureVector) -> PredictionResult:
        return self._scorer(features)


def build_rule_based_models() -> dict[PredictionTaskType, RuleBasedPredictionModel]:
    return {task_type: RuleBasedPredictionModel(task_type) for task_type in PredictionTaskType}
