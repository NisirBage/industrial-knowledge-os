from app.graph_intelligence.feature_store.store import GraphFeatureStore
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository
from app.repositories.learning_repository import SqlLearningRepository


async def _learning_repo(tmp_path):
    db_path = tmp_path / "learning_test.db"
    repository = SqlLearningRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


def _seeded_repo():
    repo = InMemoryGraphRepository()
    repo.ensure_constraints()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    valve_id = repo.create_node(EntityLabel.VALVE, "V-201", confidence=0.8)
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.6)

    inspection_id = repo.create_node(EntityLabel.INSPECTION, "inspection-1")
    repo.create_relation(pump_id, inspection_id, RelationType.INSPECTED_BY, confidence=0.5)

    incident_id = repo.create_node(EntityLabel.INCIDENT, "incident-1")
    repo.create_relation(pump_id, incident_id, RelationType.FAILED_DUE_TO, confidence=0.5)

    image_id = repo.create_node(EntityLabel.IMAGE, "img-1", properties={"source_type": "equipment_photo"})
    ocr_id = repo.create_node(EntityLabel.OCR_REGION, "ocr-1", confidence=0.9, properties={"confidence": 0.9, "text": "P-101"})
    defect_id = repo.create_node(EntityLabel.DEFECT, "defect-1", properties={"defect_type": "corrosion"})
    repo.create_relation(image_id, pump_id, RelationType.DEPICTS, confidence=0.7)
    repo.create_relation(ocr_id, pump_id, RelationType.ASSOCIATED_WITH, confidence=0.9)
    repo.create_relation(defect_id, pump_id, RelationType.ASSOCIATED_WITH, confidence=0.6)

    return repo, pump_id, valve_id


async def test_feature_definitions_lists_every_documented_feature(tmp_path):
    repo, _, _ = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    names = {f.name for f in store.feature_definitions()}
    assert names == {
        "degree", "pagerank", "betweenness", "community_id", "equipment_type",
        "maintenance_count", "incident_count", "visual_observation_count",
        "ocr_confidence", "defect_count", "prediction_confidence", "recommendation_acceptance_rate",
    }


async def test_structural_features_reflect_real_graph_topology(tmp_path):
    repo, pump_id, valve_id = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    pump_features = next(f for f in features if f.node_id == pump_id)
    valve_features = next(f for f in features if f.node_id == valve_id)
    assert pump_features.degree > valve_features.degree  # pump has more connections
    assert 0.0 <= pump_features.pagerank <= 1.0
    assert pump_features.community_id is not None


async def test_maintenance_and_incident_counts_reflect_graph_neighbors(tmp_path):
    repo, pump_id, valve_id = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    pump_features = next(f for f in features if f.node_id == pump_id)
    valve_features = next(f for f in features if f.node_id == valve_id)
    assert pump_features.maintenance_count == 1
    assert pump_features.incident_count == 1
    assert valve_features.maintenance_count == 0
    assert valve_features.incident_count == 0


async def test_visual_observation_and_defect_counts_reflect_vision_graph_neighbors(tmp_path):
    repo, pump_id, _ = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    pump_features = next(f for f in features if f.node_id == pump_id)
    assert pump_features.visual_observation_count == 2  # ocr + defect
    assert pump_features.defect_count == 1


async def test_ocr_confidence_is_averaged_from_connected_ocr_regions(tmp_path):
    repo, pump_id, _ = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    pump_features = next(f for f in features if f.node_id == pump_id)
    assert pump_features.ocr_confidence == 0.9


async def test_ocr_confidence_is_none_when_no_ocr_neighbors_exist(tmp_path):
    repo, _, valve_id = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    valve_features = next(f for f in features if f.node_id == valve_id)
    assert valve_features.ocr_confidence is None


async def test_prediction_confidence_and_recommendation_rate_come_from_learning_repository(tmp_path):
    repo, pump_id, _ = _seeded_repo()
    learning_repository = await _learning_repo(tmp_path)
    await learning_repository.record_prediction("failure_risk", "P-101", 0.6, 0.85, "explanation", "rule_based_v1")
    await learning_repository.record_feedback("recommendation_accepted", "rec-1", None, None, equipment_id="P-101")
    store = GraphFeatureStore(repo, learning_repository)

    features = await store.compute_all()

    pump_features = next(f for f in features if f.node_id == pump_id)
    assert pump_features.prediction_confidence == 0.85
    assert pump_features.recommendation_acceptance_rate == 1.0


async def test_compute_all_returns_one_row_per_graph_node(tmp_path):
    repo, _, _ = _seeded_repo()
    store = GraphFeatureStore(repo, await _learning_repo(tmp_path))

    features = await store.compute_all()

    assert len(features) == repo.node_count()
