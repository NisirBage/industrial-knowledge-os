"""Defect Detection (M7 Component 6).

Two of the eight documented defect types have a real, defensible classical-CV
detector today:

- **Corrosion/rust** (`DefectType.CORROSION`): color-signature detection —
  reddish-brown/orange hue range in HSV space, thresholded and contoured.
  `RUST` is the colloquial name for the same visual signature this detector
  produces as `CORROSION`; there is no separate rust-specific detector.
- **Cracks** (`DefectType.CRACK`): thin, elongated, high-aspect-ratio dark
  contours found via edge detection — genuinely different geometry from a
  corrosion patch (long and narrow vs. a filled blob), so it gets its own
  detector rather than reusing the color heuristic.

The other five (oil leakage, damaged insulation, loose components, missing
bolts, surface wear) are **architecture-only** — `DefectType` has the value
and `Defect`/the Vision Knowledge Graph can represent one, but no detector
function produces them, stated plainly rather than faked with an unreliable
heuristic dressed up as real detection. Each needs either a trained
classifier or a more specialized classical technique than "one color range"
or "one edge-shape rule" can honestly provide (see
docs/architecture/vision.md for what each would actually require). Adding
one later is a new function in `_DETECTORS`, not an architecture change.
"""

from typing import Callable, Protocol

import numpy as np

from app.vision.defect.models import Defect, DefectType
from app.vision.models import BoundingBox

_MIN_DEFECT_AREA_FRACTION = 0.002
_LOW_SEVERITY_THRESHOLD = 0.02
_MEDIUM_SEVERITY_THRESHOLD = 0.08

# HSV ranges for reddish-brown/orange rust/corrosion tones.
_RUST_HSV_LOWER = np.array([2, 60, 40])
_RUST_HSV_UPPER = np.array([25, 255, 200])

_MIN_CRACK_ASPECT_RATIO = 4.0
_MIN_CRACK_LENGTH = 30


class DefectDetector(Protocol):
    def detect(self, image_path, image_id: str) -> list[Defect]: ...


def _severity_for(area_fraction: float) -> str:
    if area_fraction < _LOW_SEVERITY_THRESHOLD:
        return "low"
    if area_fraction < _MEDIUM_SEVERITY_THRESHOLD:
        return "medium"
    return "high"


def _detect_corrosion(gray, hsv, frame_area: int, image_id: str) -> list[Defect]:
    import cv2

    mask = cv2.inRange(hsv, _RUST_HSV_LOWER, _RUST_HSV_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        area_fraction = area / frame_area if frame_area else 0
        if area_fraction < _MIN_DEFECT_AREA_FRACTION:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        defects.append(
            Defect(
                defect_type=DefectType.CORROSION,
                bounding_box=BoundingBox(x, y, x + w, y + h),
                confidence=round(min(0.85, 0.4 + area_fraction * 5), 4),
                image_id=image_id,
                severity=_severity_for(area_fraction),
            )
        )
    return defects


def _detect_cracks(gray, hsv, frame_area: int, image_id: str) -> list[Defect]:
    import cv2

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    defects = []
    for contour in contours:
        if len(contour) < 5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        length, thickness = max(w, h), max(1, min(w, h))
        aspect_ratio = length / thickness
        if aspect_ratio < _MIN_CRACK_ASPECT_RATIO or length < _MIN_CRACK_LENGTH:
            continue
        area_fraction = cv2.contourArea(contour) / frame_area if frame_area else 0
        defects.append(
            Defect(
                defect_type=DefectType.CRACK,
                bounding_box=BoundingBox(x, y, x + w, y + h),
                confidence=round(min(0.75, 0.3 + aspect_ratio / 20), 4),
                image_id=image_id,
                severity=_severity_for(area_fraction),
            )
        )
    return defects


_DETECTORS: dict[DefectType, Callable] = {
    DefectType.CORROSION: _detect_corrosion,
    DefectType.CRACK: _detect_cracks,
}


class ClassicalCVDefectDetector:
    name = "classical_cv_defect_detector"
    version = "1.0"

    def detect(self, image_path, image_id: str) -> list[Defect]:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return []
        frame_area = image.shape[0] * image.shape[1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        defects: list[Defect] = []
        for detector_fn in _DETECTORS.values():
            defects += detector_fn(gray, hsv, frame_area, image_id)
        return defects

    @staticmethod
    def supported_defect_types() -> list[DefectType]:
        return list(_DETECTORS)
