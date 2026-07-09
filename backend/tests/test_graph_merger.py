import pytest

from app.entity_resolution.merger import GraphMerger
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def test_merge_redirects_source_edges_onto_the_target():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump")
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_relation(duplicate_id, valve_id, RelationType.CONNECTED_TO)
    merger = GraphMerger(repo)

    merger.merge(duplicate_id, canonical_id, reason="edit_distance=0.90")

    node_ids = {n["id"] for n in repo.all_nodes()}
    assert duplicate_id not in node_ids
    assert canonical_id in node_ids
    edges = repo.all_edges()
    assert any(e["source"] == canonical_id and e["target"] == valve_id for e in edges)
    assert not any(e["source"] == duplicate_id for e in edges)


def test_merge_combines_properties_with_target_winning_on_conflict():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump", properties={"manufacturer": "Acme", "notes": "from duplicate"})
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101", properties={"notes": "canonical note"})
    merger = GraphMerger(repo)

    merger.merge(duplicate_id, canonical_id, reason="test")

    properties = repo.get_node_properties(canonical_id)
    assert properties["manufacturer"] == "Acme"  # only the duplicate had this key
    assert properties["notes"] == "canonical note"  # target's value wins on conflict


def test_merge_raises_on_self_merge():
    repo = InMemoryGraphRepository()
    node_id = repo.create_node(EntityLabel.PUMP, "P-101")
    merger = GraphMerger(repo)

    with pytest.raises(ValueError):
        merger.merge(node_id, node_id, reason="test")


def test_merge_raises_on_unknown_node_id():
    repo = InMemoryGraphRepository()
    node_id = repo.create_node(EntityLabel.PUMP, "P-101")
    merger = GraphMerger(repo)

    with pytest.raises(KeyError):
        merger.merge("does-not-exist", node_id, reason="test")


def test_merge_drops_edges_that_would_become_target_self_loops():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump")
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101")
    repo.create_relation(duplicate_id, canonical_id, RelationType.CONNECTED_TO)  # would become P-101 -> P-101
    merger = GraphMerger(repo)

    merger.merge(duplicate_id, canonical_id, reason="test")

    edges = repo.all_edges()
    assert not any(e["source"] == canonical_id and e["target"] == canonical_id for e in edges)


def test_merge_returns_a_record_with_the_original_source_snapshot():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump", confidence=0.7)
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101")
    merger = GraphMerger(repo)

    record = merger.merge(duplicate_id, canonical_id, reason="acronym_expansion=1.00")

    assert record.source_node["id"] == duplicate_id
    assert record.source_node["canonical_name"] == "Feedwater Pump"
    assert record.target_id == canonical_id
    assert record.reason == "acronym_expansion=1.00"


def test_rollback_recreates_the_source_node_and_its_original_edges():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump")
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101")
    valve_id = repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_relation(duplicate_id, valve_id, RelationType.CONNECTED_TO)
    merger = GraphMerger(repo)
    record = merger.merge(duplicate_id, canonical_id, reason="test")

    restored_id = merger.rollback(record)

    assert restored_id == duplicate_id  # entity_id_for is deterministic -> same id restored
    node_ids = {n["id"] for n in repo.all_nodes()}
    assert duplicate_id in node_ids
    edges = repo.all_edges()
    assert any(e["source"] == duplicate_id and e["target"] == valve_id for e in edges)


def test_rollback_restores_original_node_properties():
    repo = InMemoryGraphRepository()
    duplicate_id = repo.create_node(EntityLabel.PUMP, "Feedwater Pump", properties={"manufacturer": "Acme"})
    canonical_id = repo.create_node(EntityLabel.PUMP, "P-101")
    merger = GraphMerger(repo)
    record = merger.merge(duplicate_id, canonical_id, reason="test")

    merger.rollback(record)

    assert repo.get_node_properties(duplicate_id)["manufacturer"] == "Acme"
