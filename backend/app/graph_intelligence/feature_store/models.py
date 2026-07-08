"""Graph Feature Store data model (M9 Component 2)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeFeatures:
    node_id: str
    canonical_name: str
    equipment_type: str
    degree: int
    pagerank: float
    betweenness: float
    community_id: int
    maintenance_count: int
    incident_count: int
    visual_observation_count: int
    ocr_confidence: float | None
    defect_count: int
    prediction_confidence: float | None
    recommendation_acceptance_rate: float | None
