"""Feature Store (M6 Component 2): the reusable evolution of M4's
RuleBasedFeatureExtractor — same underlying computation, now with
versioned/documented feature metadata, TTL caching for online (single-
equipment, low-latency) lookups, and batch DataFrame retrieval for offline
(training-time) use. Never recomputes a feature definition's *meaning*, only
adds the metadata layer a training pipeline needs around it.
"""

import dataclasses
import time

import pandas as pd

from app.features.extractor import FeatureExtractor
from app.features.models import EquipmentFeatureVector
from app.ml.feature_store.models import FeatureDefinition

_GENERATION_TIME = "runtime"  # computed on demand, not a fixed batch timestamp

_FEATURE_DEFINITIONS: list[FeatureDefinition] = [
    FeatureDefinition("days_since_last_failure", "Days since the equipment's most recent recorded failure", "timeline_engine", "platform", "int | None", "v1", _GENERATION_TIME),
    FeatureDefinition("maintenance_frequency", "Count of maintenance-type timeline events", "timeline_engine", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("inspection_frequency", "Count of inspection-type timeline events", "timeline_engine", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("failure_count", "Count of FailureMode-typed incident records", "graph_repository.incident_history", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("incident_count", "Count of all incident_history records", "graph_repository.incident_history", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("average_repair_time", "Mean repair duration in hours", "not yet captured at ingestion", "platform", "float | None", "v1", _GENERATION_TIME),
    FeatureDefinition("recommendation_acceptance_rate", "Historical acceptance rate of recommendations for this equipment", "learning_repository.recommendation_acceptance_rate", "platform", "float | None", "v1", _GENERATION_TIME),
    FeatureDefinition("contradiction_rate", "Contradictions logged / interactions logged for this equipment", "learning_repository.contradiction_rate", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("risk_score", "Rule-derived composite of failure_count, incident_count, contradiction_rate", "features.extractor.RuleBasedFeatureExtractor", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("equipment_age", "Years since equipment install date", "not yet captured at ingestion", "platform", "int | None", "v1", _GENERATION_TIME),
    FeatureDefinition("manufacturer", "Manufacturer graph neighbor, if known", "graph_repository.find_neighbors", "platform", "str | None", "v1", _GENERATION_TIME),
    FeatureDefinition("location", "Location graph neighbor, if known", "graph_repository.find_neighbors", "platform", "str | None", "v1", _GENERATION_TIME),
    FeatureDefinition("criticality", "Rule-derived bucket (low/medium/high) from failure_count and incident_count", "features.extractor.RuleBasedFeatureExtractor", "platform", "str", "v1", _GENERATION_TIME),
    FeatureDefinition("ontology_depth", "Depth of the equipment's ontology label below IndustrialAsset", "ontology.hierarchy.ancestors_of", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("graph_degree", "Number of 1-hop graph neighbors", "graph_repository.find_neighbors", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("retrieval_frequency", "Count of chat interactions that resolved to this equipment", "learning_repository.retrieval_frequency", "platform", "int", "v1", _GENERATION_TIME),
]

_LINEAGE: dict[str, list[str]] = {
    "risk_score": ["failure_count", "incident_count", "contradiction_rate"],
    "criticality": ["failure_count", "incident_count"],
    "contradiction_rate": ["learning_repository.ContradictionRecord", "learning_repository.InteractionRecord"],
    "recommendation_acceptance_rate": ["learning_repository.FeedbackRecord"],
}


class FeatureStore:
    def __init__(self, feature_extractor: FeatureExtractor, ttl_seconds: float = 60.0):
        self._feature_extractor = feature_extractor
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, EquipmentFeatureVector]] = {}

    def feature_definitions(self) -> list[FeatureDefinition]:
        return list(_FEATURE_DEFINITIONS)

    def lineage(self, feature_name: str) -> list[str]:
        return _LINEAGE.get(feature_name, [])

    async def get_online_features(self, equipment_id: str) -> EquipmentFeatureVector:
        cached = self._cache.get(equipment_id)
        if cached is not None and (time.monotonic() - cached[0]) < self._ttl_seconds:
            return cached[1]
        features = await self._feature_extractor.extract(equipment_id)
        self._cache[equipment_id] = (time.monotonic(), features)
        return features

    async def get_offline_features(self, equipment_ids: list[str]) -> pd.DataFrame:
        rows = [dataclasses.asdict(await self.get_online_features(eid)) for eid in equipment_ids]
        return pd.DataFrame(rows)

    def invalidate(self, equipment_id: str) -> None:
        self._cache.pop(equipment_id, None)
