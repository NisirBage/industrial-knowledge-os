"""Feature Engineering Pipeline (M4 Component 2).

Combines the Knowledge Graph (structure), the Timeline Engine (event
history), and the Learning Repository (accumulated feedback/interaction
history) into one deterministic feature vector per equipment. "Deterministic"
here means: given the same graph state and the same learning-repository
state, extract() always returns the same vector — no randomness, no
model-dependent behavior (that's the Prediction Engine's job).
"""

from typing import Protocol

from app.features.models import EquipmentFeatureVector
from app.ontology.hierarchy import ancestors_of
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.repositories.learning_repository import LearningRepositoryProtocol
from app.timeline.engine import TimelineEngine
from app.timeline.models import TimelineEventType

_MANUFACTURER_TYPE = EntityLabel.MANUFACTURER.value
_LOCATION_TYPE = EntityLabel.LOCATION.value

_HIGH_CRITICALITY_FAILURE_THRESHOLD = 3
_HIGH_CRITICALITY_INCIDENT_THRESHOLD = 2
_MEDIUM_CRITICALITY_FAILURE_THRESHOLD = 1


class FeatureExtractor(Protocol):
    async def extract(self, equipment_id: str) -> EquipmentFeatureVector: ...


class RuleBasedFeatureExtractor:
    def __init__(
        self,
        graph_repository: GraphRepositoryProtocol,
        timeline_engine: TimelineEngine,
        learning_repository: LearningRepositoryProtocol,
    ):
        self._graph_repository = graph_repository
        self._timeline_engine = timeline_engine
        self._learning_repository = learning_repository

    async def extract(self, equipment_id: str) -> EquipmentFeatureVector:
        history = self._timeline_engine.full_history(equipment_id)
        maintenance_count = sum(1 for e in history if e.event_type == TimelineEventType.MAINTENANCE)
        inspection_count = sum(1 for e in history if e.event_type == TimelineEventType.INSPECTION)

        # incident_history() carries the raw ontology type per record, so
        # FAILURE_MODE (a specific failure) and INCIDENT (a broader incident
        # report, which may not be a failure — e.g. a near-miss) are counted
        # separately rather than collapsed into the single INCIDENT timeline
        # bucket TimelineEngine uses for its own (coarser) purposes.
        incident_records = self._graph_repository.incident_history(equipment_id)
        failure_count = sum(1 for r in incident_records if r["type"] == EntityLabel.FAILURE_MODE.value)
        incident_count = len(incident_records)

        node = self._graph_repository.find_entity(equipment_id)
        graph_degree = 0
        ontology_depth = 0
        manufacturer: str | None = None
        location: str | None = None

        if node is not None:
            neighbors = self._graph_repository.find_neighbors(node["id"], depth=1)
            graph_degree = len(neighbors)
            ontology_depth = len(ancestors_of(EntityLabel(node["type"]))) - 1
            manufacturer = next((n["canonical_name"] for n in neighbors if n["type"] == _MANUFACTURER_TYPE), None)
            location = next((n["canonical_name"] for n in neighbors if n["type"] == _LOCATION_TYPE), None)

        contradiction_rate = await self._learning_repository.contradiction_rate(equipment_id)
        retrieval_frequency = await self._learning_repository.retrieval_frequency(equipment_id)
        recommendation_acceptance_rate = await self._learning_repository.recommendation_acceptance_rate(equipment_id)

        risk_score = self._risk_score(failure_count, incident_count, contradiction_rate)

        return EquipmentFeatureVector(
            equipment_id=equipment_id,
            days_since_last_failure=None,  # requires temporal metadata not yet captured at ingestion
            maintenance_frequency=float(maintenance_count),
            inspection_frequency=float(inspection_count),
            failure_count=failure_count,
            incident_count=incident_count,
            average_repair_time=None,  # requires repair-duration data not yet captured at ingestion
            recommendation_acceptance_rate=recommendation_acceptance_rate,
            contradiction_rate=contradiction_rate,
            risk_score=risk_score,
            equipment_age=None,  # requires install-date metadata not yet captured at ingestion
            manufacturer=manufacturer,
            location=location,
            criticality=self._criticality(failure_count, incident_count),
            ontology_depth=max(ontology_depth, 0),
            graph_degree=graph_degree,
            retrieval_frequency=retrieval_frequency,
        )

    @staticmethod
    def _risk_score(failure_count: int, incident_count: int, contradiction_rate: float) -> float:
        raw = 0.4 * min(failure_count / 5, 1.0) + 0.3 * min(incident_count / 5, 1.0) + 0.3 * min(contradiction_rate, 1.0)
        return round(min(raw, 1.0), 4)

    @staticmethod
    def _criticality(failure_count: int, incident_count: int) -> str:
        if failure_count >= _HIGH_CRITICALITY_FAILURE_THRESHOLD or incident_count >= _HIGH_CRITICALITY_INCIDENT_THRESHOLD:
            return "high"
        if failure_count >= _MEDIUM_CRITICALITY_FAILURE_THRESHOLD:
            return "medium"
        return "low"
