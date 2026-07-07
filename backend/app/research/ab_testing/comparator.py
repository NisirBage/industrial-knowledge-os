"""A/B Evaluation Framework (M5 Component 3): paired comparison between two
"arms" (e.g. Regex vs GLiNER, Hybrid vs Graph-aware Hybrid) scored on the
same matched samples (same queries/documents evaluated by both).

The p-value uses a normal approximation to the paired t-test (via
`math.erf`, stdlib only — no scipy dependency). This is a good approximation
once there are enough paired samples (~30+); for small sample counts it's a
rough guide, not a rigorous test — documented here rather than silently
overstating precision.
"""

import math
import statistics

from app.research.ab_testing.models import ABTestResult


class ABComparator:
    def compare(
        self,
        arm_a_name: str,
        arm_b_name: str,
        metric_name: str,
        arm_a_scores: list[float],
        arm_b_scores: list[float],
    ) -> ABTestResult:
        if len(arm_a_scores) != len(arm_b_scores):
            raise ValueError("Paired A/B comparison requires equal-length score lists (matched samples).")
        if not arm_a_scores:
            raise ValueError("Cannot compare with zero samples.")

        differences = [a - b for a, b in zip(arm_a_scores, arm_b_scores, strict=True)]
        mean_diff = statistics.mean(differences)
        t_statistic, p_value = self._paired_t_test(differences)

        mean_a = statistics.mean(arm_a_scores)
        mean_b = statistics.mean(arm_b_scores)
        if mean_a > mean_b:
            winner = arm_a_name
        elif mean_b > mean_a:
            winner = arm_b_name
        else:
            winner = "tie"

        return ABTestResult(
            arm_a_name=arm_a_name,
            arm_b_name=arm_b_name,
            metric_name=metric_name,
            arm_a_scores=arm_a_scores,
            arm_b_scores=arm_b_scores,
            mean_a=round(mean_a, 4),
            mean_b=round(mean_b, 4),
            mean_delta=round(mean_diff, 4),
            t_statistic=t_statistic,
            p_value=p_value,
            winner=winner,
        )

    @staticmethod
    def _paired_t_test(differences: list[float]) -> tuple[float, float]:
        n = len(differences)
        if n < 2:
            return 0.0, 1.0
        mean_diff = statistics.mean(differences)
        std_dev = statistics.stdev(differences)
        if std_dev == 0:
            # Zero variance with a non-zero mean difference means every
            # paired sample agreed exactly — the strongest possible signal,
            # not the absence of one — so p -> 0 rather than 1.
            if mean_diff == 0:
                return 0.0, 1.0
            return (float("inf") if mean_diff > 0 else float("-inf")), 0.0

        standard_error = std_dev / math.sqrt(n)
        t_statistic = mean_diff / standard_error

        z = abs(t_statistic)
        one_sided_p = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))
        p_value = round(min(1.0, 2 * one_sided_p), 4)
        return round(t_statistic, 4), p_value
