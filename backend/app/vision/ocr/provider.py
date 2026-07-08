"""Industrial OCR (M7 Component 2). EasyOCR is a real, working choice for
this platform: pure-Python, pip-installable, downloads its detection/
recognition models over the network once (no system binary like Tesseract
requires) — the same "attempt the real thing, fall back gracefully" pattern
GLiNER/the embedder/the reranker already use. `UnavailableOCRProvider` is the
fallback if the model can't load (no network, disk quota, etc.), returning an
empty region list rather than raising — consistent with every other
degrade-not-crash dependency in this platform.

Handwritten notes are architecture-only for now: `OCRRegion` and this
Protocol have no handwriting-specific field, but nothing here assumes typed
text either — EasyOCR's recognition model does attempt handwriting, just
with lower accuracy than typed industrial labels, which is an honest
limitation to expect, not a documented non-goal.
"""

import logging
from pathlib import Path
from typing import Protocol

from app.vision.models import BoundingBox
from app.vision.ocr.models import OCRRegion

logger = logging.getLogger(__name__)


class OCRProvider(Protocol):
    def extract_text(self, image_path: str | Path, image_id: str) -> list[OCRRegion]: ...


class EasyOCRProvider:
    name = "easyocr"
    version = "1.7.2"

    def __init__(self, languages: list[str] | None = None):
        import easyocr

        self._reader = easyocr.Reader(languages or ["en"], gpu=False, verbose=False)

    def extract_text(self, image_path: str | Path, image_id: str) -> list[OCRRegion]:
        results = self._reader.readtext(str(image_path))
        regions = []
        for points, text, confidence in results:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            regions.append(
                OCRRegion(
                    text=text,
                    confidence=round(float(confidence), 4),
                    bounding_box=BoundingBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    image_id=image_id,
                )
            )
        return regions


class UnavailableOCRProvider:
    """Fallback when EasyOCR's models can't be loaded — degrades to "no OCR
    text found" rather than crashing ingestion of the image itself."""

    name = "unavailable_ocr"
    version = "1.0"

    def extract_text(self, image_path: str | Path, image_id: str) -> list[OCRRegion]:
        return []


def build_ocr_provider(languages: list[str] | None = None) -> OCRProvider:
    try:
        return EasyOCRProvider(languages)
    except Exception:
        logger.warning("EasyOCR model unavailable — falling back to no-OCR mode. Text regions will be empty.")
        return UnavailableOCRProvider()
