from app.ml.datasets.models import TASK_PROBLEM_TYPE, MLProblemType, MLTaskType


def test_every_ml_task_type_has_a_problem_type_mapping():
    assert set(TASK_PROBLEM_TYPE.keys()) == set(MLTaskType)


def test_task_problem_type_mapping_matches_documented_intent():
    assert TASK_PROBLEM_TYPE[MLTaskType.FAILURE_PREDICTION] == MLProblemType.CLASSIFICATION
    assert TASK_PROBLEM_TYPE[MLTaskType.MAINTENANCE_PRIORITY] == MLProblemType.REGRESSION
    assert TASK_PROBLEM_TYPE[MLTaskType.INSPECTION_PRIORITY] == MLProblemType.REGRESSION
    assert TASK_PROBLEM_TYPE[MLTaskType.RECOMMENDATION_ACCEPTANCE] == MLProblemType.CLASSIFICATION
    assert TASK_PROBLEM_TYPE[MLTaskType.EQUIPMENT_RISK] == MLProblemType.REGRESSION
    assert TASK_PROBLEM_TYPE[MLTaskType.COMPLIANCE_RISK] == MLProblemType.REGRESSION
    assert TASK_PROBLEM_TYPE[MLTaskType.ROOT_CAUSE_CLASSIFICATION] == MLProblemType.CLASSIFICATION
