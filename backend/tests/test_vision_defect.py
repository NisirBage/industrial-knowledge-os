import numpy as np
import pytest

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.vision.defect.detector import ClassicalCVDefectDetector
from app.vision.defect.models import DefectType

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python-headless not installed")


def _rust_patch_image(path):
    img = np.full((200, 200, 3), 200, dtype=np.uint8)  # light grey background
    cv2.rectangle(img, (50, 50), (150, 150), (14, 65, 183), -1)  # BGR rust/brown color
    cv2.imwrite(str(path), img)
    return path


def _crack_image(path):
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.line(img, (10, 100), (190, 108), (0, 0, 0), 1)  # long, thin, nearly-horizontal dark line
    cv2.imwrite(str(path), img)
    return path


def _blank_image(path):
    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.imwrite(str(path), img)
    return path


def test_detects_corrosion_colored_patch(tmp_path):
    path = _rust_patch_image(tmp_path / "rust.png")
    detector = ClassicalCVDefectDetector()

    defects = detector.detect(path, image_id="img-1")

    corrosion = [d for d in defects if d.defect_type == DefectType.CORROSION]
    assert len(corrosion) >= 1
    assert all(0.0 <= d.confidence <= 1.0 for d in corrosion)
    assert corrosion[0].severity in {"low", "medium", "high"}


def test_detects_thin_elongated_crack(tmp_path):
    path = _crack_image(tmp_path / "crack.png")
    detector = ClassicalCVDefectDetector()

    defects = detector.detect(path, image_id="img-2")

    cracks = [d for d in defects if d.defect_type == DefectType.CRACK]
    assert len(cracks) >= 1


def test_blank_image_yields_no_defects(tmp_path):
    path = _blank_image(tmp_path / "blank.png")
    detector = ClassicalCVDefectDetector()

    assert detector.detect(path, image_id="img-3") == []


def test_unreadable_path_yields_no_defects(tmp_path):
    detector = ClassicalCVDefectDetector()

    assert detector.detect(tmp_path / "missing.png", image_id="img-x") == []


def test_supported_defect_types_is_honest_subset():
    supported = ClassicalCVDefectDetector.supported_defect_types()

    assert DefectType.CORROSION in supported
    assert DefectType.CRACK in supported
    # Explicitly not implemented — architecture-only, see detector.py docstring.
    assert DefectType.OIL_LEAKAGE not in supported
    assert DefectType.MISSING_BOLT not in supported


def test_all_defect_types_are_documented_in_the_enum():
    expected = {
        "corrosion", "rust", "oil_leakage", "crack",
        "damaged_insulation", "loose_component", "missing_bolt", "surface_wear",
    }
    assert {t.value for t in DefectType} == expected
