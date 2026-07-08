import json
from pathlib import Path

import networkx as nx
import torch

from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def _seeded_repo() -> InMemoryGraphRepository:
    repo = InMemoryGraphRepository()
    repo.ensure_constraints()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    valve_id = repo.create_node(EntityLabel.VALVE, "V-201", confidence=0.8)
    motor_id = repo.create_node(EntityLabel.MOTOR, "M-301", confidence=0.7)
    repo.create_relation(pump_id, valve_id, RelationType.CONNECTED_TO, confidence=0.6)
    repo.create_relation(motor_id, pump_id, RelationType.PART_OF, confidence=0.5)
    return repo


def test_build_returns_all_nodes_and_edges():
    dataset = GraphDatasetBuilder(_seeded_repo()).build()

    assert len(dataset.nodes) == 3
    assert len(dataset.edges) == 2
    assert dataset.node_types == sorted([EntityLabel.PUMP.value, EntityLabel.VALVE.value, EntityLabel.MOTOR.value])


def test_build_is_deterministic_across_repeated_calls():
    repo = _seeded_repo()
    builder = GraphDatasetBuilder(repo)

    first = builder.build()
    second = builder.build()

    assert first.node_id_to_index == second.node_id_to_index
    assert [n.id for n in first.nodes] == [n.id for n in second.nodes]


def test_node_id_to_index_matches_sorted_id_order():
    dataset = GraphDatasetBuilder(_seeded_repo()).build()

    sorted_ids = sorted(n.id for n in dataset.nodes)
    assert [dataset.node_id_to_index[nid] for nid in sorted_ids] == list(range(len(sorted_ids)))


def test_to_networkx_handles_a_node_whose_properties_dict_itself_has_a_confidence_key():
    """Regression test: vision nodes (M7) commonly store their own
    `confidence` inside `properties` (see
    VisionKnowledgeGraphBuilder._node_properties()) — to_networkx must not
    collide that with the node's first-class confidence field."""
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.OCR_REGION, "ocr-1", confidence=0.9, properties={"confidence": 0.5, "text": "P-101"})
    builder = GraphDatasetBuilder(repo)
    dataset = builder.build()

    graph = builder.to_networkx(dataset)

    node_id = dataset.nodes[0].id
    assert graph.nodes[node_id]["confidence"] == 0.9
    assert graph.nodes[node_id]["properties"]["confidence"] == 0.5


def test_to_networkx_produces_a_multidigraph_with_matching_node_and_edge_counts():
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()

    graph = builder.to_networkx(dataset)

    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2


def test_to_pyg_homogeneous_produces_correct_shapes():
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()

    data = builder.to_pyg_homogeneous(dataset)

    assert data.num_nodes == 3
    assert data.edge_index.shape == (2, 2)
    assert data.x.shape[0] == 3


def test_to_pyg_homogeneous_accepts_custom_node_features():
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()
    custom_features = torch.rand((3, 5))

    data = builder.to_pyg_homogeneous(dataset, node_features=custom_features)

    assert torch.equal(data.x, custom_features)


def test_to_pyg_homogeneous_handles_a_graph_with_no_edges():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-999")
    builder = GraphDatasetBuilder(repo)
    dataset = builder.build()

    data = builder.to_pyg_homogeneous(dataset)

    assert data.edge_index.shape == (2, 0)


def test_to_pyg_heterogeneous_creates_one_node_type_block_per_type():
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()

    hetero = builder.to_pyg_heterogeneous(dataset)

    assert set(hetero.node_types) == {EntityLabel.PUMP.value, EntityLabel.VALVE.value, EntityLabel.MOTOR.value}


def test_to_pyg_heterogeneous_creates_edge_blocks_keyed_by_type_triple():
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()

    hetero = builder.to_pyg_heterogeneous(dataset)

    edge_types = set(hetero.edge_types)
    assert (EntityLabel.PUMP.value, RelationType.CONNECTED_TO.value, EntityLabel.VALVE.value) in edge_types
    assert (EntityLabel.MOTOR.value, RelationType.PART_OF.value, EntityLabel.PUMP.value) in edge_types


def test_to_csv_writes_nodes_and_edges_files(tmp_path):
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"

    builder.to_csv(dataset, nodes_path, edges_path)

    nodes_text = nodes_path.read_text(encoding="utf-8")
    edges_text = edges_path.read_text(encoding="utf-8")
    assert "P-101" in nodes_text
    assert "canonical_name" in nodes_text
    assert len(edges_text.strip().splitlines()) == 3  # header + 2 edges


def test_to_graphml_writes_a_valid_graphml_file(tmp_path):
    builder = GraphDatasetBuilder(_seeded_repo())
    dataset = builder.build()
    path = tmp_path / "graph.graphml"

    builder.to_graphml(dataset, path)

    reloaded = nx.read_graphml(path)
    assert reloaded.number_of_nodes() == 3
    assert reloaded.number_of_edges() == 2


def test_to_json_writes_nodes_and_edges_with_properties(tmp_path):
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101", properties={"model": "LMV-311"})
    builder = GraphDatasetBuilder(repo)
    dataset = builder.build()
    path = tmp_path / "graph.json"

    builder.to_json(dataset, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["nodes"]) == 1
    assert payload["nodes"][0]["properties"]["model"] == "LMV-311"


def test_multimodal_node_types_are_exported_side_by_side_with_text_nodes():
    """M9 spec: 'multimodal node types' should just work, since vision nodes
    (M7) are ordinary graph nodes with no special-cased export path."""
    repo = InMemoryGraphRepository()
    pump_id = repo.create_node(EntityLabel.PUMP, "P-101")
    image_id = repo.create_node(EntityLabel.IMAGE, "img-1", properties={"width": 200})
    repo.create_relation(image_id, pump_id, RelationType.DEPICTS, confidence=0.7)

    dataset = GraphDatasetBuilder(repo).build()

    assert EntityLabel.IMAGE.value in dataset.node_types
    assert EntityLabel.PUMP.value in dataset.node_types
    assert dataset.relation_types == [RelationType.DEPICTS.value]
