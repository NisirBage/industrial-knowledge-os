"""Training Dataset Builder data model (M6 Component 1)."""

from enum import Enum


class MLTaskType(str, Enum):
    FAILURE_PREDICTION = "failure_prediction"
    MAINTENANCE_PRIORITY = "maintenance_priority"
    INSPECTION_PRIORITY = "inspection_priority"
    RECOMMENDATION_ACCEPTANCE = "recommendation_acceptance"
    EQUIPMENT_RISK = "equipment_risk"
    COMPLIANCE_RISK = "compliance_risk"
    ROOT_CAUSE_CLASSIFICATION = "root_cause_classification"


class MLProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    GRAPH = "graph"


# Which problem type each task naturally is — used so callers (and the ML
# Lab API) don't have to know this mapping themselves. Ranking/time-series/
# graph tasks aren't concretely demonstrated with a synthetic label
# generator in this milestone (see synthetic.py docstring) but the
# classification/regression tasks are fully real, not placeholders.
TASK_PROBLEM_TYPE: dict[MLTaskType, MLProblemType] = {
    MLTaskType.FAILURE_PREDICTION: MLProblemType.CLASSIFICATION,
    MLTaskType.MAINTENANCE_PRIORITY: MLProblemType.REGRESSION,
    MLTaskType.INSPECTION_PRIORITY: MLProblemType.REGRESSION,
    MLTaskType.RECOMMENDATION_ACCEPTANCE: MLProblemType.CLASSIFICATION,
    MLTaskType.EQUIPMENT_RISK: MLProblemType.REGRESSION,
    MLTaskType.COMPLIANCE_RISK: MLProblemType.REGRESSION,
    MLTaskType.ROOT_CAUSE_CLASSIFICATION: MLProblemType.CLASSIFICATION,
}
