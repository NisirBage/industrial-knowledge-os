"""Defect Detection data model (M7 Component 6)."""

from dataclasses import dataclass
from enum import Enum

from app.vision.models import BoundingBox


class DefectType(str, Enum):
    CORROSION = "corrosion"
    RUST = "rust"
    OIL_LEAKAGE = "oil_leakage"
    CRACK = "crack"
    DAMAGED_INSULATION = "damaged_insulation"
    LOOSE_COMPONENT = "loose_component"
    MISSING_BOLT = "missing_bolt"
    SURFACE_WEAR = "surface_wear"


@dataclass(frozen=True)
class Defect:
    defect_type: DefectType
    bounding_box: BoundingBox
    confidence: float
    image_id: str
    severity: str  # "low" | "medium" | "high", derived from affected area coverage
