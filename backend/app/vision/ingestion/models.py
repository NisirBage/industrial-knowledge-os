"""Vision Ingestion data model (M7 Component 1)."""

from dataclasses import dataclass
from enum import Enum


class VisionSourceType(str, Enum):
    INSPECTION_PHOTO = "inspection_photo"
    EQUIPMENT_PHOTO = "equipment_photo"
    PID_DIAGRAM = "pid_diagram"
    ELECTRICAL_SCHEMATIC = "electrical_schematic"
    ENGINEERING_DRAWING = "engineering_drawing"
    MAINTENANCE_IMAGE = "maintenance_image"
    HANDWRITTEN_NOTE = "handwritten_note"
    # Future, documented but not implemented — see docs/roadmap/README.md:
    # thermal imaging, drone imagery, CCTV snapshots. Adding one is a new
    # enum value plus a source-specific analysis hint, not a pipeline rewrite.


# Which source types a diagram-shaped parse (symbols/lines) vs. a photo-shaped
# analysis (defects/gauges/equipment blobs) makes sense for — used to route
# an ingested image to the right downstream analyzers without guessing from
# pixels alone (classifying "is this a P&ID or a photo" from content is out
# of scope for the classical-CV baseline; the uploader states it).
DIAGRAM_SOURCE_TYPES = frozenset({VisionSourceType.PID_DIAGRAM, VisionSourceType.ELECTRICAL_SCHEMATIC, VisionSourceType.ENGINEERING_DRAWING})
PHOTO_SOURCE_TYPES = frozenset({VisionSourceType.INSPECTION_PHOTO, VisionSourceType.EQUIPMENT_PHOTO, VisionSourceType.MAINTENANCE_IMAGE})


@dataclass(frozen=True)
class VisionDocument:
    id: str
    filename: str
    source_type: VisionSourceType
    content_type: str
    width: int
    height: int
    storage_path: str
    size_bytes: int
    equipment_hint: str | None = None
