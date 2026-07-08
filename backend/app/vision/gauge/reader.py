"""Gauge & Instrument Reading (M7 Component 5).

Deliberately separated into four stages — detection, reading, validation,
knowledge-graph mapping — so a future ML-based needle reader can replace
only `read_gauge()` (e.g. a regression model predicting normalized_value
directly from the cropped dial image) while `detect_gauges()`,
`validate_reading()`, and the graph-mapping caller stay unchanged. This is
the same swap-in-one-stage discipline `PredictionEngine.register_model()`
established in M4/M6, applied to vision.

Reading assumes the common 270-degree industrial dial convention: minimum at
-135 degrees from top (lower-left, "7:30"), maximum at +135 degrees
(lower-right, "4:30"), sweeping clockwise through the top. This produces a
normalized 0.0-1.0 fraction of full scale, not a calibrated engineering unit
— no min/max label is available to convert "73% of scale" into "146 PSI."
Digital displays are architecture-only for now: `GaugeType` has the value,
but reading one would need OCR on a segmented digit region, which the OCR
subsystem (Component 2) can supply once wired together as an extension.
"""

import math
from enum import Enum

from app.vision.gauge.models import DetectedGauge, GaugeReading
from app.vision.models import BoundingBox

_MIN_ANGLE_DEGREES = -135.0
_MAX_ANGLE_DEGREES = 135.0
_MIN_GAUGE_RADIUS = 15
_MAX_GAUGE_RADIUS = 150


class GaugeType(str, Enum):
    ANALOG_DIAL = "analog_dial"
    DIGITAL_DISPLAY = "digital_display"  # architecture only — see module docstring


def detect_gauges(image_path, image_id: str) -> list[DetectedGauge]:
    """Stage 1: locate candidate circular dial faces via Hough Circle
    Transform. Deliberately the same technique as the diagram parser's valve
    detection but tuned for photo-scale dials (larger radius range) — the
    two are kept separate methods, not shared, because a P&ID valve symbol
    and a real photographed gauge face have different expected sizes/noise
    characteristics, and conflating them would make both harder to tune."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
        param1=50, param2=35, minRadius=_MIN_GAUGE_RADIUS, maxRadius=_MAX_GAUGE_RADIUS,
    )
    if circles is None:
        return []

    gauges = []
    for x, y, r in circles[0].reshape(-1, 3):
        x, y, r = int(x), int(y), int(r)
        gauges.append(
            DetectedGauge(
                bounding_box=BoundingBox(x - r, y - r, x + r, y + r),
                image_id=image_id,
                center=(x, y),
                radius=r,
            )
        )
    return gauges


def read_gauge(image_path, gauge: DetectedGauge) -> GaugeReading:
    """Stage 2: within the detected dial's bounding box, find the needle as
    the longest line segment with one endpoint near the dial center, and
    compute its angle from vertical (top = 0 degrees, clockwise positive)."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return GaugeReading(gauge.bounding_box, gauge.image_id, None, None, 0.0, False, "Image unreadable.")

    x_min, y_min, x_max, y_max = gauge.bounding_box.x_min, gauge.bounding_box.y_min, gauge.bounding_box.x_max, gauge.bounding_box.y_max
    x_min, y_min = max(0, x_min), max(0, y_min)
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return GaugeReading(gauge.bounding_box, gauge.image_id, None, None, 0.0, False, "Empty region of interest.")

    center_x, center_y = gauge.center[0] - x_min, gauge.center[1] - y_min
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=20, minLineLength=max(10, gauge.radius // 3), maxLineGap=5)

    if lines is None:
        return GaugeReading(gauge.bounding_box, gauge.image_id, None, None, 0.0, False, "No needle-like line segment detected.")

    needle_candidates = []
    center_tolerance = max(6, gauge.radius // 4)
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dist1 = math.hypot(x1 - center_x, y1 - center_y)
        dist2 = math.hypot(x2 - center_x, y2 - center_y)
        near_x, near_y, near_dist, far_x, far_y = (x1, y1, dist1, x2, y2) if dist1 < dist2 else (x2, y2, dist2, x1, y1)
        if near_dist > center_tolerance:
            continue
        length = math.hypot(far_x - near_x, far_y - near_y)
        needle_candidates.append((length, far_x, far_y))

    if not needle_candidates:
        return GaugeReading(gauge.bounding_box, gauge.image_id, None, None, 0.0, False, "No line segment passes near the dial center.")

    _, far_x, far_y = max(needle_candidates, key=lambda c: c[0])
    dx, dy = far_x - center_x, far_y - center_y
    angle_from_top = math.degrees(math.atan2(dx, -dy))  # 0 = up, +90 = right, -90 = left, +-180 = down

    value, valid, note = _angle_to_normalized_value(angle_from_top)
    confidence = 0.6 if valid else 0.3
    return GaugeReading(gauge.bounding_box, gauge.image_id, round(angle_from_top, 2), value, confidence, valid, note)


def validate_reading(reading: GaugeReading) -> GaugeReading:
    """Stage 3: a no-op pass-through today (the angle-to-value conversion
    already validates range in read_gauge) — kept as its own function/stage
    per the spec's explicit architecture requirement, and the seam a future
    validator (e.g. cross-checking against a prior reading of the same
    gauge for physically-implausible jumps) would extend."""
    return reading


def _angle_to_normalized_value(angle_from_top: float) -> tuple[float | None, bool, str]:
    if angle_from_top < _MIN_ANGLE_DEGREES or angle_from_top > _MAX_ANGLE_DEGREES:
        return None, False, (
            f"Needle angle {angle_from_top:.1f} degrees is outside the standard 270-degree dial "
            f"sweep ({_MIN_ANGLE_DEGREES} to {_MAX_ANGLE_DEGREES} degrees from top) — reading not trusted."
        )
    value = (angle_from_top - _MIN_ANGLE_DEGREES) / (_MAX_ANGLE_DEGREES - _MIN_ANGLE_DEGREES)
    return round(value, 4), True, "OK"


class GaugeReader:
    """Stage 4 (composition): detect -> read -> validate for every gauge
    found in an image. `detect_gauges`/`read_gauge`/`validate_reading` remain
    free functions so a future implementation can reuse any subset of them
    directly without going through this class."""

    name = "classical_cv_gauge_reader"
    version = "1.0"

    def read_all(self, image_path, image_id: str) -> list[GaugeReading]:
        readings = []
        for gauge in detect_gauges(image_path, image_id):
            readings.append(validate_reading(read_gauge(image_path, gauge)))
        return readings
