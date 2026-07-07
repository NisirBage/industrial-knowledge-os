"""Model Monitoring (M6 Component 7).

Prediction/feature drift use the Population Stability Index (PSI) — a
standard, pure-Python/math drift metric (no scipy needed): bin the baseline
distribution into deciles, compare how much of the recent distribution falls
into each bin. PSI < 0.1 is conventionally "no meaningful shift," 0.1-0.25
"moderate," > 0.25 "significant" — those thresholds are encoded in
`drift_severity` rather than left for callers to guess at.

Concept drift and calibration reuse LearningRepository (M4) and Explainer
(M6) rather than re-implementing accuracy tracking or calibration curves
here — monitoring's job is to interpret those numbers over time, not
recompute them. Latency/memory are already captured by
PredictionEngine.predict() via record_model_performance (M4) and the
Offline Evaluation Harness's tracemalloc/process_time instrumentation (M5);
this module doesn't duplicate that.
"""

import math

from app.repositories.learning_repository import LearningRepositoryProtocol

_NO_DRIFT_THRESHOLD = 0.1
_MODERATE_DRIFT_THRESHOLD = 0.25
_EPSILON = 1e-6


def _quantile_bin_edges(values: list[float], n_bins: int) -> list[float]:
    sorted_values = sorted(values)
    n = len(sorted_values)
    edges = [sorted_values[0]]
    for i in range(1, n_bins):
        index = min(int(i * n / n_bins), n - 1)
        edges.append(sorted_values[index])
    edges.append(sorted_values[-1])

    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge != deduped[-1]:
            deduped.append(edge)
    return deduped


def _bin_counts(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) - 1)
    last_bin_index = len(edges) - 2
    for value in values:
        for i in range(len(edges) - 1):
            is_last_bin = i == last_bin_index
            if edges[i] <= value < edges[i + 1] or (is_last_bin and value == edges[i + 1]):
                counts[i] += 1
                break
    return counts


def population_stability_index(baseline: list[float], recent: list[float], n_bins: int = 10) -> float:
    if not baseline or not recent:
        return 0.0

    edges = _quantile_bin_edges(baseline, n_bins)
    if len(edges) < 3:
        return 0.0  # not enough distinct baseline values to bin meaningfully

    baseline_counts = _bin_counts(baseline, edges)
    recent_counts = _bin_counts(recent, edges)
    baseline_total = sum(baseline_counts) or 1
    recent_total = sum(recent_counts) or 1

    psi = 0.0
    for baseline_count, recent_count in zip(baseline_counts, recent_counts, strict=True):
        baseline_pct = max(baseline_count / baseline_total, _EPSILON)
        recent_pct = max(recent_count / recent_total, _EPSILON)
        psi += (recent_pct - baseline_pct) * math.log(recent_pct / baseline_pct)

    return round(psi, 4)


class ModelMonitor:
    def __init__(self, learning_repository: LearningRepositoryProtocol):
        self._learning_repository = learning_repository

    def prediction_drift(self, baseline_predictions: list[float], recent_predictions: list[float]) -> float:
        return population_stability_index(baseline_predictions, recent_predictions)

    def feature_drift(self, baseline_values: list[float], recent_values: list[float]) -> float:
        return population_stability_index(baseline_values, recent_values)

    @staticmethod
    def drift_severity(psi: float) -> str:
        if psi < _NO_DRIFT_THRESHOLD:
            return "none"
        if psi < _MODERATE_DRIFT_THRESHOLD:
            return "moderate"
        return "significant"

    async def concept_drift(self, metric_name: str = "accuracy") -> float | None:
        """Current average of a tracked accuracy-style metric across all
        recorded evaluations. A true trend (is it *getting* worse) needs
        windowed time comparison once enough evaluation history
        accumulates — this surfaces the current baseline that a future
        windowed comparison would diff against."""
        return await self._learning_repository.average_model_accuracy(metric_name)

    async def recommendation_acceptance_trend(self, equipment_id: str) -> float | None:
        return await self._learning_repository.recommendation_acceptance_rate(equipment_id)
