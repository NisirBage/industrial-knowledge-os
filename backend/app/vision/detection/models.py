"""Equipment Recognition data model (M7 Component 4)."""

from dataclasses import dataclass

from app.vision.models import BoundingBox


@dataclass(frozen=True)
class Detection:
    equipment_type: str
    bounding_box: BoundingBox
    confidence: float
    image_reference: str
    ontology_label: str | None = None
    # Populated once VisionKnowledgeGraphBuilder links this detection to a
    # graph node (its own DetectedObject id, or the ASSOCIATED_WITH entity's
    # id if an equipment_hint resolved it) — None immediately after detect().
    equipment_id: str | None = None
