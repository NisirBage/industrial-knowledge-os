from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def test_create_node_is_idempotent_and_merges_aliases():
    repo = InMemoryGraphRepository()

    first_id = repo.create_node(EntityLabel.PUMP, "P-101", aliases=frozenset({"Pump P-101"}), confidence=0.6)
    second_id = repo.create_node(EntityLabel.PUMP, "P-101", aliases=frozenset({"Feedwater Pump"}), confidence=0.9)

    assert first_id == second_id
    node = repo.find_entity("P-101", EntityLabel.PUMP)
    assert set(node["aliases"]) == {"Pump P-101", "Feedwater Pump"}


def test_create_node_stores_and_get_node_properties_retrieves_arbitrary_properties():
    """Regression test: create_node's `properties` argument must actually be
    retrievable — the Neo4j-backed implementation previously accepted the
    parameter but silently dropped it (never persisted to the database),
    an unexercised bug until M7's vision graph builder became the first
    real caller relying on it."""
    repo = InMemoryGraphRepository()

    node_id = repo.create_node(EntityLabel.IMAGE, "img-1", properties={"width": 640, "height": 480, "filename": "gauge.png"})

    properties = repo.get_node_properties(node_id)
    assert properties["width"] == 640
    assert properties["height"] == 480
    assert properties["filename"] == "gauge.png"


def test_get_node_properties_returns_empty_dict_for_unknown_node():
    repo = InMemoryGraphRepository()

    assert repo.get_node_properties("does-not-exist") == {}


def test_all_nodes_returns_every_created_node_with_its_fields():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.8, properties={"foo": "bar"})
    valve_id = repo.create_node(EntityLabel.VALVE, "V-201", confidence=0.6)

    nodes = repo.all_nodes()

    ids = {n["id"] for n in nodes}
    assert ids == {pump_id, valve_id}
    pump_node = next(n for n in nodes if n["id"] == pump_id)
    assert pump_node["type"] == EntityLabel.PUMP.value
    assert pump_node["canonical_name"] == "P-101"
    assert pump_node["confidence"] == 0.8
    assert pump_node["properties"] == {"foo": "bar"}


def test_all_edges_returns_every_created_relation():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-201")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.7)

    edges = repo.all_edges()

    assert len(edges) == 1
    assert edges[0]["source"] == pump_id
    assert edges[0]["target"] == valve_id
    assert edges[0]["relation_type"] == RelationType.CONNECTED_TO.value
    assert edges[0]["confidence"] == 0.7


def test_all_nodes_and_all_edges_are_empty_for_a_fresh_repository():
    repo = InMemoryGraphRepository()

    assert repo.all_nodes() == []
    assert repo.all_edges() == []


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
