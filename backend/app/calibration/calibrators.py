"""Calibration methods (O3 Component 2): fit a real transform from raw
confidence scores to calibrated probabilities, given (confidence, correct)
labeled pairs — no fabricated correction factors.
"""

import math
from typing import Protocol

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPSILON = 1e-6


class Calibrator(Protocol):
    def fit(self, confidences: list[float], correctness: list[bool]) -> None: ...
    def calibrate(self, confidences: list[float]) -> list[float]: ...


def _clip(p: float) -> float:
    return min(max(p, _EPSILON), 1 - _EPSILON)


def _logit(p: float) -> float:
    p = _clip(p)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class TemperatureScalingCalibrator:
    """Fits a single scalar temperature T minimizing negative log-likelihood
    of calibrated = sigmoid(logit(p) / T) against the true labels — the
    standard single-parameter temperature scaling method (Guo et al. 2017),
    generalized here to arbitrary confidence scores rather than only
    softmax logits."""

    def __init__(self):
        self._temperature = 1.0

    def fit(self, confidences: list[float], correctness: list[bool]) -> None:
        logits = [_logit(p) for p in confidences]
        labels = [1.0 if c else 0.0 for c in correctness]

        def negative_log_likelihood(temperature: float) -> float:
            if temperature <= 0:
                return float("inf")
            total = 0.0
            for logit, label in zip(logits, labels, strict=True):
                calibrated = _clip(_sigmoid(logit / temperature))
                total -= label * math.log(calibrated) + (1 - label) * math.log(1 - calibrated)
            return total

        result = minimize_scalar(negative_log_likelihood, bounds=(0.05, 20.0), method="bounded")
        self._temperature = float(result.x)

    def calibrate(self, confidences: list[float]) -> list[float]:
        return [round(_sigmoid(_logit(p) / self._temperature), 4) for p in confidences]

    @property
    def temperature(self) -> float:
        return self._temperature


class IsotonicRegressionCalibrator:
    """Wraps `sklearn.isotonic.IsotonicRegression` — a real, non-parametric
    monotonic fit from raw confidence to calibrated probability."""

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, confidences: list[float], correctness: list[bool]) -> None:
        labels = [1.0 if c else 0.0 for c in correctness]
        self._model.fit(np.array(confidences), np.array(labels))

    def calibrate(self, confidences: list[float]) -> list[float]:
        return [round(float(v), 4) for v in self._model.predict(np.array(confidences))]


class BetaCalibrator:
    """Real (simplified) Beta calibration (Kull et al. 2017): fits
    calibrated = sigmoid(a*log(p) + b*log(1-p) + c) via logistic regression
    on the two log-odds features — the standard linearization of the Beta
    calibration map, using `sklearn.linear_model.LogisticRegression` rather
    than a from-scratch Beta-distribution MLE."""

    def __init__(self):
        self._model = LogisticRegression()

    def fit(self, confidences: list[float], correctness: list[bool]) -> None:
        features = [[math.log(_clip(p)), math.log(1 - _clip(p))] for p in confidences]
        labels = [1 if c else 0 for c in correctness]
        if len(set(labels)) < 2:
            # LogisticRegression requires both classes; fall back to identity
            # rather than raising or fabricating a fit on degenerate data.
            self._model = None
            return
        self._model.fit(np.array(features), np.array(labels))

    def calibrate(self, confidences: list[float]) -> list[float]:
        if self._model is None:
            return [round(p, 4) for p in confidences]
        features = [[math.log(_clip(p)), math.log(1 - _clip(p))] for p in confidences]
        return [round(float(v), 4) for v in self._model.predict_proba(np.array(features))[:, 1]]
