"""Engineering Diagram Understanding data model (M7 Component 3)."""

from dataclasses import dataclass
from enum import Enum

from app.vision.models import BoundingBox


class SymbolType(str, Enum):
    VALVE = "valve"
    PUMP = "pump"
    PIPE = "pipe"
    SENSOR = "sensor"
    FLOW_ARROW = "flow_arrow"
    INSTRUMENT = "instrument"
    LABEL = "label"


@dataclass(frozen=True)
class DiagramSymbol:
    symbol_type: SymbolType
    bounding_box: BoundingBox
    confidence: float
    image_id: str
