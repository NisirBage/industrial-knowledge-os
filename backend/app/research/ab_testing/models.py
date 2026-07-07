from dataclasses import dataclass


@dataclass(frozen=True)
class ABTestResult:
    arm_a_name: str
    arm_b_name: str
    metric_name: str
    arm_a_scores: list[float]
    arm_b_scores: list[float]
    mean_a: float
    mean_b: float
    mean_delta: float
    t_statistic: float
    p_value: float
    winner: str
