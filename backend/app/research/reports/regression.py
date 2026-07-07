"""Regression detection (M5 Report Generation): compares a new
BenchmarkReport against the most recent prior report for the same benchmark
name, flagging any metric that moved the wrong direction by more than a
threshold.

Direction-aware: most metrics are "higher is better" (precision, recall,
coverage, ...), but a few are "lower is better" (latency, error, cost,
hallucination). Matched by substring against a known list — a metric name
this doesn't recognize is treated as higher-is-better by default, which is
the more common case.
"""

from dataclasses import dataclass

from app.research.benchmarks.models import BenchmarkReport

_LOWER_IS_BETTER_SUBSTRINGS = ("latency", "error", "hallucination", "cost", "duration")

_DEFAULT_THRESHOLD = 0.05


@dataclass(frozen=True)
class RegressionFinding:
    metric_name: str
    previous_value: float
    current_value: float
    percent_change: float
    is_regression: bool


def _is_lower_better(metric_name: str) -> bool:
    return any(token in metric_name for token in _LOWER_IS_BETTER_SUBSTRINGS)


class RegressionDetector:
    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        self._threshold = threshold

    def detect(self, previous: BenchmarkReport, current: BenchmarkReport) -> list[RegressionFinding]:
        previous_by_name = {metric.name: metric.value for metric in previous.metrics}
        findings: list[RegressionFinding] = []

        for metric in current.metrics:
            if metric.name not in previous_by_name:
                continue
            previous_value = previous_by_name[metric.name]
            if previous_value == 0:
                continue

            percent_change = round((metric.value - previous_value) / abs(previous_value), 4)
            if _is_lower_better(metric.name):
                is_regression = percent_change > self._threshold
            else:
                is_regression = percent_change < -self._threshold

            findings.append(
                RegressionFinding(
                    metric_name=metric.name,
                    previous_value=previous_value,
                    current_value=metric.value,
                    percent_change=percent_change,
                    is_regression=is_regression,
                )
            )

        return findings

    @staticmethod
    def has_regression(findings: list[RegressionFinding]) -> bool:
        return any(f.is_regression for f in findings)
