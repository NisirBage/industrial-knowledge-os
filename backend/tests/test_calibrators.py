from app.calibration.calibrators import BetaCalibrator, IsotonicRegressionCalibrator, TemperatureScalingCalibrator
from app.calibration.metrics import expected_calibration_error


def _overconfident_dataset():
    # Confidence always 0.95 but only half are actually correct: a clean,
    # deterministic overconfidence case every calibrator should reduce.
    confidences = [0.95] * 20
    correctness = [True] * 10 + [False] * 10
    return confidences, correctness


def test_temperature_scaling_reduces_overconfidence():
    confidences, correctness = _overconfident_dataset()
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(confidences, correctness)

    calibrated = calibrator.calibrate(confidences)

    assert expected_calibration_error(calibrated, correctness) < expected_calibration_error(confidences, correctness)
    assert calibrator.temperature > 1.0  # temperature > 1 flattens overconfident probabilities toward 0.5


def test_temperature_scaling_calibrate_returns_values_in_unit_interval():
    confidences, correctness = _overconfident_dataset()
    calibrator = TemperatureScalingCalibrator()
    calibrator.fit(confidences, correctness)

    calibrated = calibrator.calibrate([0.1, 0.5, 0.99])
    assert all(0.0 <= v <= 1.0 for v in calibrated)


def test_isotonic_regression_reduces_overconfidence():
    confidences, correctness = _overconfident_dataset()
    calibrator = IsotonicRegressionCalibrator()
    calibrator.fit(confidences, correctness)

    calibrated = calibrator.calibrate(confidences)

    assert expected_calibration_error(calibrated, correctness) < expected_calibration_error(confidences, correctness)


def test_isotonic_regression_is_monotonic():
    calibrator = IsotonicRegressionCalibrator()
    calibrator.fit([0.1, 0.3, 0.5, 0.7, 0.9] * 4, [False, False, True, True, True] * 4)

    calibrated = calibrator.calibrate([0.1, 0.3, 0.5, 0.7, 0.9])
    assert calibrated == sorted(calibrated)


def test_beta_calibrator_reduces_overconfidence():
    confidences, correctness = _overconfident_dataset()
    calibrator = BetaCalibrator()
    calibrator.fit(confidences, correctness)

    calibrated = calibrator.calibrate(confidences)

    assert expected_calibration_error(calibrated, correctness) < expected_calibration_error(confidences, correctness)


def test_beta_calibrator_falls_back_to_identity_when_only_one_class_present():
    calibrator = BetaCalibrator()
    calibrator.fit([0.8, 0.9, 0.95], [True, True, True])  # degenerate: only one label

    calibrated = calibrator.calibrate([0.8, 0.9, 0.95])
    assert calibrated == [0.8, 0.9, 0.95]
