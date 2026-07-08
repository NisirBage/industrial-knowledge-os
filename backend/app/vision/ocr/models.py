"""Industrial OCR data model (M7 Component 2). Every OCR hit becomes a
structured OCRRegion — text, confidence, and pixel coordinates — never a
flat text blob, so downstream consumers (the Vision Knowledge Graph, the
Vision Workspace's overlay) can place it precisely on the source image."""

from dataclasses import dataclass

from app.vision.models import BoundingBox

__all__ = ["BoundingBox", "OCRRegion"]


@dataclass(frozen=True)
class OCRRegion:
    text: str
    confidence: float
    bounding_box: BoundingBox
    image_id: str
