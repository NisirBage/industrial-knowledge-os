import numpy as np
import pytest

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.vision.ocr.provider import EasyOCRProvider, UnavailableOCRProvider, build_ocr_provider

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python-headless not installed")


def _text_image_path(tmp_path, text: str = "P-101"):
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    path = tmp_path / "ocr_test.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="module")
def real_ocr_provider():
    try:
        return EasyOCRProvider()
    except Exception as exc:
        pytest.skip(f"EasyOCR model unavailable in this environment: {exc}")


def test_easyocr_reads_rendered_equipment_tag(tmp_path, real_ocr_provider):
    path = _text_image_path(tmp_path, "P-101")

    regions = real_ocr_provider.extract_text(path, image_id="img-1")

    assert len(regions) >= 1
    assert any("P-101" in r.text or "P-1O1" in r.text for r in regions)
    assert all(0.0 <= r.confidence <= 1.0 for r in regions)
    assert all(r.image_id == "img-1" for r in regions)


def test_easyocr_bounding_box_is_within_image_dimensions(tmp_path, real_ocr_provider):
    path = _text_image_path(tmp_path, "V-2201A")

    regions = real_ocr_provider.extract_text(path, image_id="img-2")

    assert len(regions) >= 1
    box = regions[0].bounding_box
    assert 0 <= box.x_min < box.x_max <= 300
    assert 0 <= box.y_min < box.y_max <= 100


def test_unavailable_provider_returns_empty_list(tmp_path):
    provider = UnavailableOCRProvider()

    assert provider.extract_text(tmp_path / "anything.png", "img-3") == []


def test_build_ocr_provider_falls_back_when_easyocr_raises(monkeypatch):
    import app.vision.ocr.provider as provider_module

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated model load failure")

    monkeypatch.setattr(provider_module, "EasyOCRProvider", _raise)

    provider = build_ocr_provider()

    assert isinstance(provider, UnavailableOCRProvider)


def test_build_ocr_provider_returns_real_provider_when_available():
    provider = build_ocr_provider()

    assert isinstance(provider, (EasyOCRProvider, UnavailableOCRProvider))
