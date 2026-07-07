from app.features.extractor import RuleBasedFeatureExtractor
from app.timeline.engine import TimelineEngine


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        return [
            {"id": "n1", "type": "Manufacturer", "canonical_name": "Acme Pumps", "relation_type": "RELATED_TO"},
            {"id": "n2", "type": "Location", "canonical_name": "Zone 4", "relation_type": "LOCATED_IN"},
            {"id": "n3", "type": "Standard", "canonical_name": "ISO 9001", "relation_type": "USES_STANDARD"},
        ]

    def maintenance_history(self, canonical_name: str):
        return [
            {"id": "m1", "type": "CorrectiveAction", "canonical_name": "Bearing Replacement", "relation_type": "MAINTAINED_BY"},
            {"id": "m2", "type": "Inspection", "canonical_name": "Q1 Inspection", "relation_type": "INSPECTED_BY"},
        ]

    def incident_history(self, canonical_name: str):
        return [
            {"id": "i1", "type": "FailureMode", "canonical_name": "Bearing Wear", "relation_type": "FAILED_DUE_TO"},
            {"id": "i2", "type": "Incident", "canonical_name": "Near Miss", "relation_type": "RELATED_TO"},
        ]


class _FakeLearningRepository:
    def __init__(self, contradiction_rate=0.0, retrieval_frequency=0, acceptance_rate=None):
        self._contradiction_rate = contradiction_rate
        self._retrieval_frequency = retrieval_frequency
        self._acceptance_rate = acceptance_rate

    async def contradiction_rate(self, equipment_id: str) -> float:
        return self._contradiction_rate

    async def retrieval_frequency(self, equipment_id: str) -> int:
        return self._retrieval_frequency

    async def recommendation_acceptance_rate(self, equipment_id: str):
        return self._acceptance_rate


async def test_extract_computes_failure_and_incident_counts_separately():
    extractor = RuleBasedFeatureExtractor(_FakeGraphRepository(), TimelineEngine(_FakeGraphRepository()), _FakeLearningRepository())

    features = await extractor.extract("P-101")

    assert features.failure_count == 1  # only the FailureMode-typed record
    assert features.incident_count == 2  # both incident_history records
    assert features.maintenance_frequency == 1.0
    assert features.inspection_frequency == 1.0


def _repo():
    return _FakeGraphRepository()


async def test_extract_finds_manufacturer_and_location_from_graph_neighbors():
    extractor = RuleBasedFeatureExtractor(_repo(), TimelineEngine(_repo()), _FakeLearningRepository())
    features = await extractor.extract("P-101")

    assert features.manufacturer == "Acme Pumps"
    assert features.location == "Zone 4"


async def test_extract_returns_none_for_fields_not_yet_captured_by_ingestion():
    extractor = RuleBasedFeatureExtractor(_repo(), TimelineEngine(_repo()), _FakeLearningRepository())
    features = await extractor.extract("P-101")

    assert features.days_since_last_failure is None
    assert features.average_repair_time is None
    assert features.equipment_age is None


async def test_extract_propagates_learning_repository_rates():
    extractor = RuleBasedFeatureExtractor(
        _repo(), TimelineEngine(_repo()), _FakeLearningRepository(contradiction_rate=0.5, retrieval_frequency=7, acceptance_rate=0.8)
    )
    features = await extractor.extract("P-101")

    assert features.contradiction_rate == 0.5
    assert features.retrieval_frequency == 7
    assert features.recommendation_acceptance_rate == 0.8


async def test_extract_unknown_equipment_has_zero_graph_degree_and_depth():
    extractor = RuleBasedFeatureExtractor(_repo(), TimelineEngine(_repo()), _FakeLearningRepository())
    features = await extractor.extract("UNKNOWN-999")

    assert features.graph_degree == 0
    assert features.ontology_depth == 0
    assert features.manufacturer is None


async def test_criticality_is_high_when_failure_threshold_met():
    class _HighFailureGraphRepository(_FakeGraphRepository):
        def incident_history(self, canonical_name: str):
            return [{"id": f"f{i}", "type": "FailureMode", "canonical_name": f"Failure {i}", "relation_type": "FAILED_DUE_TO"} for i in range(3)]

    extractor = RuleBasedFeatureExtractor(_HighFailureGraphRepository(), TimelineEngine(_HighFailureGraphRepository()), _FakeLearningRepository())
    features = await extractor.extract("P-101")

    assert features.criticality == "high"


async def test_criticality_is_low_with_no_history():
    class _NoHistoryGraphRepository(_FakeGraphRepository):
        def incident_history(self, canonical_name: str):
            return []

        def maintenance_history(self, canonical_name: str):
            return []

    repo = _NoHistoryGraphRepository()
    extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), _FakeLearningRepository())
    features = await extractor.extract("P-101")

    assert features.criticality == "low"
    assert features.risk_score == 0.0


async def test_ontology_depth_reflects_hierarchy_position():
    extractor = RuleBasedFeatureExtractor(_repo(), TimelineEngine(_repo()), _FakeLearningRepository())
    features = await extractor.extract("P-101")

    # Pump -> RotatingEquipment -> IndustrialAsset: depth 2
    assert features.ontology_depth == 2
