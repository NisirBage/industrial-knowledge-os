import asyncio

from app.features.models import EquipmentFeatureVector
from app.ml.feature_store.store import FeatureStore


def _feature_vector(equipment_id: str) -> EquipmentFeatureVector:
    return EquipmentFeatureVector(
        equipment_id=equipment_id,
        days_since_last_failure=5,
        maintenance_frequency=1.0,
        inspection_frequency=1.0,
        failure_count=1,
        incident_count=1,
        average_repair_time=None,
        recommendation_acceptance_rate=None,
        contradiction_rate=0.0,
        risk_score=0.3,
        equipment_age=None,
        manufacturer=None,
        location=None,
        criticality="low",
        ontology_depth=1,
        graph_degree=2,
        retrieval_frequency=0,
    )


class _CountingFeatureExtractor:
    def __init__(self):
        self.calls = 0

    async def extract(self, equipment_id: str) -> EquipmentFeatureVector:
        self.calls += 1
        return _feature_vector(equipment_id)


def test_feature_definitions_returns_sixteen_documented_features():
    store = FeatureStore(_CountingFeatureExtractor())
    definitions = store.feature_definitions()

    assert len(definitions) == 16
    names = {d.name for d in definitions}
    assert "risk_score" in names
    assert "criticality" in names
    for definition in definitions:
        assert definition.description
        assert definition.source
        assert definition.owner
        assert definition.datatype
        assert definition.version
        assert definition.generation_time


def test_lineage_returns_known_dependencies():
    store = FeatureStore(_CountingFeatureExtractor())
    assert store.lineage("risk_score") == ["failure_count", "incident_count", "contradiction_rate"]


def test_lineage_returns_empty_list_for_unknown_feature():
    store = FeatureStore(_CountingFeatureExtractor())
    assert store.lineage("not_a_real_feature") == []


async def test_get_online_features_caches_within_ttl():
    extractor = _CountingFeatureExtractor()
    store = FeatureStore(extractor, ttl_seconds=60.0)

    await store.get_online_features("P-101")
    await store.get_online_features("P-101")

    assert extractor.calls == 1


async def test_get_online_features_recomputes_after_ttl_expires():
    extractor = _CountingFeatureExtractor()
    store = FeatureStore(extractor, ttl_seconds=0.05)

    await store.get_online_features("P-101")
    await asyncio.sleep(0.1)
    await store.get_online_features("P-101")

    assert extractor.calls == 2


async def test_invalidate_forces_recomputation():
    extractor = _CountingFeatureExtractor()
    store = FeatureStore(extractor, ttl_seconds=60.0)

    await store.get_online_features("P-101")
    store.invalidate("P-101")
    await store.get_online_features("P-101")

    assert extractor.calls == 2


async def test_get_online_features_caches_independently_per_equipment():
    extractor = _CountingFeatureExtractor()
    store = FeatureStore(extractor, ttl_seconds=60.0)

    await store.get_online_features("P-101")
    await store.get_online_features("V-201")

    assert extractor.calls == 2


async def test_get_offline_features_returns_one_row_per_equipment_id():
    store = FeatureStore(_CountingFeatureExtractor())
    df = await store.get_offline_features(["P-101", "P-102", "P-103"])

    assert len(df) == 3
    assert set(df["equipment_id"]) == {"P-101", "P-102", "P-103"}
