"""Calibration & Uncertainty metrics (O3 Component 2).

Every function takes parallel `confidences` (predicted probability the
outcome was correct, in [0, 1]) and `correctness` (the real observed
outcome, True/False) — real paired data, never fabricated. `brier_score`
reuses `sklearn.metrics.brier_score_loss`, the same function already used
for classical model evaluation in `app/ml/training/trainer.py`, rather than
reimplementing it a second time.
"""

from sklearn.metrics import brier_score_loss

from app.calibration.models import CalibrationReport, HistogramBucket, ReliabilityBucket

_DEFAULT_BINS = 10


def _bucket_index(confidence: float, n_bins: int) -> int:
    return min(int(confidence * n_bins), n_bins - 1)


def _bucket_range_label(index: int, n_bins: int) -> str:
    low = round(index * 100 / n_bins)
    high = round((index + 1) * 100 / n_bins)
    return f"{low}-{high}%"


def reliability_buckets(confidences: list[float], correctness: list[bool], n_bins: int = _DEFAULT_BINS) -> list[ReliabilityBucket]:
    if len(confidences) != len(correctness):
        raise ValueError("confidences and correctness must be the same length")

    bucket_confidences: list[list[float]] = [[] for _ in range(n_bins)]
    bucket_correct: list[list[bool]] = [[] for _ in range(n_bins)]
    for confidence, correct in zip(confidences, correctness, strict=True):
        index = _bucket_index(confidence, n_bins)
        bucket_confidences[index].append(confidence)
        bucket_correct[index].append(correct)

    buckets = []
    for index in range(n_bins):
        count = len(bucket_confidences[index])
        mean_confidence = round(sum(bucket_confidences[index]) / count, 4) if count else 0.0
        accuracy = round(sum(bucket_correct[index]) / count, 4) if count else None
        buckets.append(ReliabilityBucket(_bucket_range_label(index, n_bins), mean_confidence, accuracy, count))
    return buckets


def confidence_histogram(confidences: list[float], n_bins: int = _DEFAULT_BINS) -> list[HistogramBucket]:
    counts = [0] * n_bins
    for confidence in confidences:
        counts[_bucket_index(confidence, n_bins)] += 1
    return [HistogramBucket(_bucket_range_label(i, n_bins), counts[i]) for i in range(n_bins)]


def expected_calibration_error(confidences: list[float], correctness: list[bool], n_bins: int = _DEFAULT_BINS) -> float:
    """ECE = sum over bins of (|bin| / N) * |accuracy(bin) - mean_confidence(bin)|."""
    if not confidences:
        return 0.0
    buckets = reliability_buckets(confidences, correctness, n_bins)
    total = len(confidences)
    error = sum(
        (bucket.count / total) * abs(bucket.accuracy - bucket.mean_confidence)
        for bucket in buckets
        if bucket.count > 0 and bucket.accuracy is not None
    )
    return round(error, 4)


def maximum_calibration_error(confidences: list[float], correctness: list[bool], n_bins: int = _DEFAULT_BINS) -> float:
    """MCE = max over non-empty bins of |accuracy(bin) - mean_confidence(bin)|."""
    if not confidences:
        return 0.0
    buckets = reliability_buckets(confidences, correctness, n_bins)
    gaps = [abs(bucket.accuracy - bucket.mean_confidence) for bucket in buckets if bucket.count > 0 and bucket.accuracy is not None]
    return round(max(gaps), 4) if gaps else 0.0


def brier_score(confidences: list[float], correctness: list[bool]) -> float:
    if not confidences:
        return 0.0
    return round(float(brier_score_loss([1.0 if c else 0.0 for c in correctness], confidences)), 4)


def expected_calibration_error_from_buckets(buckets: list[ReliabilityBucket]) -> float:
    """Same ECE formula as `expected_calibration_error`, computed directly
    from already-bucketed (mean_confidence, accuracy, count) data — for
    calibration sources (like `LearningRepository.confidence_calibration_buckets()`)
    that expose real recorded outcomes pre-aggregated into buckets rather
    than raw per-prediction pairs."""
    total = sum(b.count for b in buckets)
    if not total:
        return 0.0
    error = sum((b.count / total) * abs(b.accuracy - b.mean_confidence) for b in buckets if b.count > 0 and b.accuracy is not None)
    return round(error, 4)


def maximum_calibration_error_from_buckets(buckets: list[ReliabilityBucket]) -> float:
    gaps = [abs(b.accuracy - b.mean_confidence) for b in buckets if b.count > 0 and b.accuracy is not None]
    return round(max(gaps), 4) if gaps else 0.0


def evaluate_calibration(confidences: list[float], correctness: list[bool], n_bins: int = _DEFAULT_BINS) -> CalibrationReport:
    return CalibrationReport(
        expected_calibration_error=expected_calibration_error(confidences, correctness, n_bins),
        maximum_calibration_error=maximum_calibration_error(confidences, correctness, n_bins),
        brier_score=brier_score(confidences, correctness),
        reliability_buckets=reliability_buckets(confidences, correctness, n_bins),
        confidence_histogram=confidence_histogram(confidences, n_bins),
        sample_count=len(confidences),
    )
