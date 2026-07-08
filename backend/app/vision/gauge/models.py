"""Gauge & Instrument Reading data model (M7 Component 5)."""

from dataclasses import dataclass

from app.vision.models import BoundingBox


@dataclass(frozen=True)
class DetectedGauge:
    bounding_box: BoundingBox
    image_id: str
    center: tuple[int, int]
    radius: int


@dataclass(frozen=True)
class GaugeReading:
    bounding_box: BoundingBox
    image_id: str
    needle_angle_degrees: float | None
    # Fraction of full scale (0.0-1.0), assuming a standard 270-degree
    # industrial dial sweep — NOT a calibrated engineering unit (PSI, °C,
    # ...): that requires a user-provided min/max scale label this baseline
    # doesn't have. See docs/architecture/vision.md for the honest scope.
    normalized_value: float | None
    confidence: float
    valid: bool
    validation_note: str
