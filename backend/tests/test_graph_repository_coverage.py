from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def test_distinct_entity_types_reflects_created_nodes():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101")
    repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_node(EntityLabel.PUMP, "P-102")

    assert repo.distinct_entity_types() == {"Pump", "Valve"}


def test_distinct_relation_types_reflects_created_edges():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO)
    repo.create_relation(pump_id, valve_id, RelationType.MENTIONS)

    assert repo.distinct_relation_types() == {"CONNECTED_TO", "MENTIONS"}


def test_node_count_and_edge_count():
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO)

    assert repo.node_count() == 2
    assert repo.edge_count() == 1


def test_empty_graph_has_no_types_and_zero_counts():
    repo = InMemoryGraphRepository()
    assert repo.distinct_entity_types() == set()
    assert repo.distinct_relation_types() == set()
    assert repo.node_count() == 0
    assert repo.edge_count() == 0
