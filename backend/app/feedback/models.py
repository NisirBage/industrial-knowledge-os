from dataclasses import dataclass
from enum import Enum


class FeedbackKind(str, Enum):
    RECOMMENDATION_ACCEPTED = "recommendation_accepted"
    RECOMMENDATION_REJECTED = "recommendation_rejected"
    RECOMMENDATION_MODIFIED = "recommendation_modified"
    ENTITY_CORRECTION = "entity_correction"
    RELATION_CORRECTION = "relation_correction"
    RECOMMENDATION_CORRECTION = "recommendation_correction"


@dataclass(frozen=True)
class Feedback:
    id: str
    kind: FeedbackKind
    target_id: str
    equipment_id: str | None
    reason: str | None
    corrected_value: str | None
