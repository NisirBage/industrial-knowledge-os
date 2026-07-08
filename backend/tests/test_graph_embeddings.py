from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.embeddings.gnn import GATEmbedding, GCNEmbedding, GraphSAGEEmbedding
from app.graph_intelligence.embeddings.node2vec import DeepWalkEmbedding, Node2VecEmbedding
from app.graph_intelligence.embeddings.operators import HadamardEdgeEmbedding, MeanPoolingGraphEmbedding
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def _seeded_dataset():
    repo = InMemoryGraphRepository()
    repo.ensure_constraints()
    ids = []
    for i in range(6):
        ids.append(repo.create_node(EntityLabel.PUMP, f"P-{i}"))
    for i in range(len(ids) - 1):
        repo.create_relation(ids[i], ids[i + 1], RelationType.CONNECTED_TO, confidence=0.5)
    repo.create_relation(ids[-1], ids[0], RelationType.CONNECTED_TO, confidence=0.5)
    return GraphDatasetBuilder(repo).build()


def test_node2vec_produces_one_embedding_per_node_with_correct_dimensions():
    dataset = _seeded_dataset()
    model = Node2VecEmbedding(dimensions=8)

    result = model.fit(dataset)

    assert len(result.embeddings) == len(dataset.nodes)
    assert all(len(e.vector) == 8 for e in result.embeddings)
    assert result.model_name == "node2vec"


def test_node2vec_is_deterministic_for_a_fixed_seed():
    dataset = _seeded_dataset()
    model_a = Node2VecEmbedding(dimensions=8, seed=7)
    model_b = Node2VecEmbedding(dimensions=8, seed=7)

    result_a = model_a.fit(dataset)
    result_b = model_b.fit(dataset)

    vector_a = result_a.vector_for(dataset.nodes[0].id)
    vector_b = result_b.vector_for(dataset.nodes[0].id)
    assert vector_a == vector_b


def test_deepwalk_is_node2vec_with_unbiased_walk_parameters():
    model = DeepWalkEmbedding()

    assert model._p == 1.0
    assert model._q == 1.0
    assert model.name == "deepwalk"


def test_deepwalk_produces_embeddings_for_every_node():
    dataset = _seeded_dataset()
    model = DeepWalkEmbedding(dimensions=8)

    result = model.fit(dataset)

    assert len(result.embeddings) == len(dataset.nodes)


def test_node2vec_handles_an_empty_graph():
    from app.graph_intelligence.dataset.models import GraphDataset

    empty_dataset = GraphDataset(nodes=[], edges=[], node_id_to_index={})
    model = Node2VecEmbedding()

    result = model.fit(empty_dataset)

    assert result.embeddings == []


def test_gcn_embedding_produces_correct_shape_and_reports_training_time():
    dataset = _seeded_dataset()
    model = GCNEmbedding(dimensions=8, hidden_dimensions=16, epochs=5)

    result = model.fit(dataset)

    assert len(result.embeddings) == len(dataset.nodes)
    assert all(len(e.vector) == 8 for e in result.embeddings)
    assert result.training_seconds >= 0.0
    assert result.model_name == "gcn"


def test_graphsage_embedding_produces_correct_shape():
    dataset = _seeded_dataset()
    model = GraphSAGEEmbedding(dimensions=8, hidden_dimensions=16, epochs=5)

    result = model.fit(dataset)

    assert len(result.embeddings) == len(dataset.nodes)
    assert result.model_name == "graphsage"


def test_gat_embedding_produces_correct_shape():
    dataset = _seeded_dataset()
    model = GATEmbedding(dimensions=8, hidden_dimensions=16, epochs=5)

    result = model.fit(dataset)

    assert len(result.embeddings) == len(dataset.nodes)
    assert result.model_name == "gat"


def test_gnn_embedding_handles_an_empty_graph():
    from app.graph_intelligence.dataset.models import GraphDataset

    empty_dataset = GraphDataset(nodes=[], edges=[], node_id_to_index={})
    model = GCNEmbedding(epochs=2)

    result = model.fit(empty_dataset)

    assert result.embeddings == []


def test_gnn_embedding_handles_a_graph_with_nodes_but_no_edges():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-only")
    dataset = GraphDatasetBuilder(repo).build()
    model = GCNEmbedding(dimensions=4, epochs=2)

    result = model.fit(dataset)

    assert len(result.embeddings) == 1


def test_hadamard_edge_embedding_multiplies_elementwise():
    embedder = HadamardEdgeEmbedding()

    result = embedder.embed_edge([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])

    assert result == [2.0, 4.0, 6.0]


def test_mean_pooling_graph_embedding_averages_vectors():
    embedder = MeanPoolingGraphEmbedding()

    result = embedder.embed_graph([[1.0, 1.0], [3.0, 5.0]])

    assert result == [2.0, 3.0]


def test_mean_pooling_graph_embedding_handles_empty_input():
    embedder = MeanPoolingGraphEmbedding()

    assert embedder.embed_graph([]) == []
