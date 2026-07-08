import math

import numpy as np
import pytest

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.vision.gauge.models import DetectedGauge, GaugeReading
from app.vision.gauge.reader import GaugeReader, _angle_to_normalized_value, detect_gauges, read_gauge, validate_reading
from app.vision.models import BoundingBox

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python-headless not installed")

_CENTER = (150, 150)
_RADIUS = 90


def _gauge_image(path, needle_angle_degrees: float, needle_length_fraction: float = 0.75):
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(img, _CENTER, _RADIUS, (0, 0, 0), 3)
    length = _RADIUS * needle_length_fraction
    angle_rad = math.radians(needle_angle_degrees)
    dx = length * math.sin(angle_rad)
    dy = -length * math.cos(angle_rad)
    endpoint = (int(_CENTER[0] + dx), int(_CENTER[1] + dy))
    cv2.line(img, _CENTER, endpoint, (0, 0, 0), 4)
    cv2.imwrite(str(path), img)
    return path


def test_detect_gauges_finds_the_dial(tmp_path):
    path = _gauge_image(tmp_path / "gauge.png", 0.0)

    gauges = detect_gauges(path, image_id="img-1")

    assert len(gauges) >= 1
    assert all(g.image_id == "img-1" for g in gauges)


def test_needle_pointing_straight_up_reads_as_half_scale(tmp_path):
    path = _gauge_image(tmp_path / "gauge.png", 0.0)
    gauge = DetectedGauge(BoundingBox(_CENTER[0] - _RADIUS, _CENTER[1] - _RADIUS, _CENTER[0] + _RADIUS, _CENTER[1] + _RADIUS), "img-1", _CENTER, _RADIUS)

    reading = read_gauge(path, gauge)

    assert reading.valid
    assert reading.normalized_value == pytest.approx(0.5, abs=0.05)


def test_needle_pointing_lower_left_reads_as_near_zero(tmp_path):
    path = _gauge_image(tmp_path / "gauge.png", -130.0)
    gauge = DetectedGauge(BoundingBox(_CENTER[0] - _RADIUS, _CENTER[1] - _RADIUS, _CENTER[0] + _RADIUS, _CENTER[1] + _RADIUS), "img-1", _CENTER, _RADIUS)

    reading = read_gauge(path, gauge)

    assert reading.valid
    assert reading.normalized_value < 0.1


def test_needle_pointing_lower_right_reads_as_near_full_scale(tmp_path):
    path = _gauge_image(tmp_path / "gauge.png", 130.0)
    gauge = DetectedGauge(BoundingBox(_CENTER[0] - _RADIUS, _CENTER[1] - _RADIUS, _CENTER[0] + _RADIUS, _CENTER[1] + _RADIUS), "img-1", _CENTER, _RADIUS)

    reading = read_gauge(path, gauge)

    assert reading.valid
    assert reading.normalized_value > 0.9


def test_angle_beyond_dial_sweep_is_invalid():
    value, valid, note = _angle_to_normalized_value(170.0)

    assert value is None
    assert not valid
    assert "outside" in note


def test_angle_at_exact_boundaries_is_valid():
    assert _angle_to_normalized_value(-135.0)[1] is True
    assert _angle_to_normalized_value(135.0)[1] is True
    assert _angle_to_normalized_value(-135.0)[0] == 0.0
    assert _angle_to_normalized_value(135.0)[0] == 1.0


def test_validate_reading_is_pass_through():
    original = GaugeReading(BoundingBox(0, 0, 10, 10), "img-1", 0.0, 0.5, 0.6, True, "OK")

    validated = validate_reading(original)

    assert validated == original


def test_reader_reads_all_gauges_in_image(tmp_path):
    path = _gauge_image(tmp_path / "gauge.png", 0.0)
    reader = GaugeReader()

    readings = reader.read_all(path, image_id="img-1")

    assert len(readings) >= 1


def test_unreadable_image_yields_no_gauges(tmp_path):
    assert detect_gauges(tmp_path / "missing.png", image_id="img-x") == []
