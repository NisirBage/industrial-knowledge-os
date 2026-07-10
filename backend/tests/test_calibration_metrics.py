from app.calibration.metrics import (
    brier_score,
    confidence_histogram,
    evaluate_calibration,
    expected_calibration_error,
    expected_calibration_error_from_buckets,
    maximum_calibration_error,
    maximum_calibration_error_from_buckets,
    reliability_buckets,
)
from app.calibration.models import ReliabilityBucket


def test_perfectly_calibrated_predictions_have_zero_ece():
    # 10 items at confidence 0.9, 9 correct -> bucket accuracy 0.9 == mean confidence 0.9
    confidences = [0.9] * 10
    correctness = [True] * 9 + [False]
    assert expected_calibration_error(confidences, correctness) == 0.0


def test_overconfident_predictions_have_positive_ece():
    confidences = [0.95] * 10
    correctness = [True] * 5 + [False] * 5  # actual accuracy 0.5, way below confidence
    assert expected_calibration_error(confidences, correctness) > 0.3


def test_mce_is_at_least_as_large_as_ece():
    confidences = [0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
    correctness = [True, True, True, False, False, False, True, True]
    ece = expected_calibration_error(confidences, correctness)
    mce = maximum_calibration_error(confidences, correctness)
    assert mce >= ece


def test_brier_score_zero_for_perfect_confident_predictions():
    assert brier_score([1.0, 1.0, 0.0], [True, True, False]) == 0.0


def test_brier_score_positive_for_wrong_confident_predictions():
    assert brier_score([1.0, 1.0], [False, False]) == 1.0


def test_reliability_buckets_cover_all_samples():
    confidences = [0.05, 0.15, 0.55, 0.95]
    correctness = [True, False, True, True]
    buckets = reliability_buckets(confidences, correctness, n_bins=10)
    assert sum(b.count for b in buckets) == 4
    assert len(buckets) == 10


def test_reliability_bucket_with_no_samples_has_none_accuracy():
    buckets = reliability_buckets([0.95], [True], n_bins=10)
    empty_bucket = buckets[0]
    assert empty_bucket.count == 0
    assert empty_bucket.accuracy is None


def test_confidence_histogram_counts_match_input_length():
    confidences = [0.1, 0.2, 0.9, 0.95, 0.5]
    histogram = confidence_histogram(confidences, n_bins=10)
    assert sum(b.count for b in histogram) == len(confidences)


def test_empty_input_produces_zero_metrics_not_a_crash():
    assert expected_calibration_error([], []) == 0.0
    assert maximum_calibration_error([], []) == 0.0
    assert brier_score([], []) == 0.0


def test_evaluate_calibration_returns_full_report():
    confidences = [0.9] * 10
    correctness = [True] * 9 + [False]
    report = evaluate_calibration(confidences, correctness)

    assert report.expected_calibration_error == 0.0
    assert report.sample_count == 10
    assert len(report.reliability_buckets) == 10
    assert len(report.confidence_histogram) == 10
    assert 0.0 <= report.brier_score <= 1.0


def test_ece_from_buckets_matches_ece_from_raw_pairs():
    confidences = [0.9, 0.9, 0.9, 0.1, 0.1]
    correctness = [True, True, False, False, True]
    raw_ece = expected_calibration_error(confidences, correctness)

    buckets = reliability_buckets(confidences, correctness)
    bucket_ece = expected_calibration_error_from_buckets(buckets)

    assert bucket_ece == raw_ece


def test_mce_from_buckets_matches_mce_from_raw_pairs():
    confidences = [0.9, 0.9, 0.9, 0.1, 0.1]
    correctness = [True, True, False, False, True]
    raw_mce = maximum_calibration_error(confidences, correctness)

    buckets = reliability_buckets(confidences, correctness)
    bucket_mce = maximum_calibration_error_from_buckets(buckets)

    assert bucket_mce == raw_mce


def test_ece_from_buckets_handles_empty_bucket_list():
    assert expected_calibration_error_from_buckets([]) == 0.0
    assert maximum_calibration_error_from_buckets([]) == 0.0


def test_ece_from_buckets_ignores_buckets_with_no_accuracy():
    buckets = [
        ReliabilityBucket("0-10%", 0.05, None, 0),
        ReliabilityBucket("90-100%", 0.95, 0.95, 10),
    ]
    assert expected_calibration_error_from_buckets(buckets) == 0.0


def test_mismatched_lengths_raise_value_error():
    import pytest

    with pytest.raises(ValueError):
        reliability_buckets([0.5, 0.6], [True])
