import numpy as np
import pytest

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.vision.detection.detector import ClassicalCVObjectDetector

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python-headless not installed")


def _equipment_photo(path):
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    cv2.rectangle(img, (80, 80), (220, 220), (30, 30, 30), -1)  # dark "equipment" blob
    cv2.imwrite(str(path), img)
    return path


def test_detects_salient_foreground_region(tmp_path):
    path = _equipment_photo(tmp_path / "equipment.png")
    detector = ClassicalCVObjectDetector()

    detections = detector.detect(path, image_id="img-1")

    assert len(detections) >= 1
    assert all(d.image_reference == "img-1" for d in detections)
    assert all(0.0 <= d.confidence <= 1.0 for d in detections)


def test_largest_region_tagged_with_equipment_hint(tmp_path):
    path = _equipment_photo(tmp_path / "equipment.png")
    detector = ClassicalCVObjectDetector()

    detections = detector.detect(path, image_id="img-1", equipment_hint="P-101")

    assert detections[0].equipment_type == "P-101"


def test_detections_without_hint_are_unknown_type(tmp_path):
    path = _equipment_photo(tmp_path / "equipment.png")
    detector = ClassicalCVObjectDetector()

    detections = detector.detect(path, image_id="img-1")

    assert all(d.equipment_type == "unknown" for d in detections)


def test_detection_equipment_id_defaults_to_none(tmp_path):
    path = _equipment_photo(tmp_path / "equipment.png")
    detector = ClassicalCVObjectDetector()

    detections = detector.detect(path, image_id="img-1")

    assert all(d.equipment_id is None for d in detections)


def test_blank_image_yields_no_detections(tmp_path):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((200, 200, 3), 255, dtype=np.uint8))
    detector = ClassicalCVObjectDetector()

    assert detector.detect(path, image_id="img-blank") == []


def test_unreadable_path_yields_no_detections(tmp_path):
    detector = ClassicalCVObjectDetector()

    assert detector.detect(tmp_path / "missing.png", image_id="img-x") == []
