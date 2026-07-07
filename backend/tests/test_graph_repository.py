from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def test_create_node_is_idempotent_and_merges_aliases():
    repo = InMemoryGraphRepository()

    first_id = repo.create_node(EntityLabel.PUMP, "P-101", aliases=frozenset({"Pump P-101"}), confidence=0.6)
    second_id = repo.create_node(EntityLabel.PUMP, "P-101", aliases=frozenset({"Feedwater Pump"}), confidence=0.9)

    assert first_id == second_id
    node = repo.find_entity("P-101", EntityLabel.PUMP)
    assert set(node["aliases"]) == {"Pump P-101", "Feedwater Pump"}


def test_create_relation_accumulates_observation_count():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-2201A")

    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.5, chunk_id="chunk-1")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.8, chunk_id="chunk-2")

    edge = repo._edges[(pump_id, valve_id, RelationType.CONNECTED_TO.value)]
    assert edge["observation_count"] == 2
    assert edge["confidence"] == 0.8
    assert edge["chunk_ids"] == {"chunk-1", "chunk-2"}


def test_find_neighbors_respects_relation_type_filter():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-2201A")
    standard_id = repo.create_node(EntityLabel.STANDARD, "ISO 9001")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO)
    repo.create_relation(pump_id, standard_id, RelationType.USES_STANDARD)

    all_neighbors = repo.find_neighbors(pump_id)
    connected_only = repo.find_neighbors(pump_id, relation_type=RelationType.CONNECTED_TO)

    assert {n["canonical_name"] for n in all_neighbors} == {"V-2201A", "ISO 9001"}
    assert {n["canonical_name"] for n in connected_only} == {"V-2201A"}


def test_equipment_history_returns_only_maintenance_descendant_nodes():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-2201A")
    incident_id = repo.create_node(EntityLabel.INCIDENT, "Bearing Failure 2024")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO)
    repo.create_relation(pump_id, incident_id, RelationType.FAILED_DUE_TO)

    history = repo.equipment_history("P-101")

    assert {h["canonical_name"] for h in history} == {"Bearing Failure 2024"}


def test_documents_mentioning_and_entities_for_document_are_inverse():
    repo = InMemoryGraphRepository()
    document_node_id = repo.upsert_document("doc-1", "manual.pdf")
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    repo.create_relation(document_node_id, pump_id, RelationType.MENTIONS, confidence=1.0, chunk_id="chunk-1")

    documents = repo.documents_mentioning("P-101")
    entities = repo.entities_for_document("doc-1")

    assert documents == [{"document_node_id": document_node_id, "document_id": "doc-1", "filename": "manual.pdf"}]
    assert {e["canonical_name"] for e in entities} == {"P-101"}


def test_subgraph_includes_seed_and_one_hop_neighbors():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-2201A")
    unrelated_id = repo.create_node(EntityLabel.LOCATION, "Zone 9")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO)

    result = repo.subgraph([pump_id], depth=1)
    node_ids = {n["id"] for n in result["nodes"]}

    assert pump_id in node_ids
    assert valve_id in node_ids
    assert unrelated_id not in node_ids
    assert len(result["edges"]) == 1
