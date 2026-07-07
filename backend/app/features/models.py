"""Equipment feature vector — the strongly typed input every Prediction
Model consumes. Fields that require data the platform doesn't capture yet
(equipment_age, manufacturer, average_repair_time — no ingestion pipeline
extracts install dates or repair durations today) are honestly Optional
rather than filled with a fabricated default; a rule-based model that needs
one of these must handle it being None the same way a human analyst would
handle a blank field on a work order.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EquipmentFeatureVector:
    equipment_id: str
    days_since_last_failure: int | None
    maintenance_frequency: float
    inspection_frequency: float
    failure_count: int
    incident_count: int
    average_repair_time: float | None
    recommendation_acceptance_rate: float | None
    contradiction_rate: float
    risk_score: float
    equipment_age: int | None
    manufacturer: str | None
    location: str | None
    criticality: str
    ontology_depth: int
    graph_degree: int
    retrieval_frequency: int
