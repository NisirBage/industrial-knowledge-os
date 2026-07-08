from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.embeddings.node2vec import Node2VecEmbedding
from app.graph_intelligence.link_prediction import LinkPredictionTrainer, LinkPredictor
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository


def _seeded_dataset_and_repo():
    repo = InMemoryGraphRepository()
    ids = [repo.create_node(EntityLabel.PUMP, f"P-{i}") for i in range(8)]
    for i in range(len(ids) - 1):
        repo.create_relation(ids[i], ids[i + 1], RelationType.CONNECTED_TO, confidence=0.6)
    repo.create_relation(ids[-1], ids[0], RelationType.CONNECTED_TO, confidence=0.6)
    dataset = GraphDatasetBuilder(repo).build()
    return repo, dataset, ids


def test_link_prediction_trainer_produces_a_fitted_classifier():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)

    model = LinkPredictionTrainer().fit(embeddings, dataset)

    assert hasattr(model.classifier, "classes_")


def test_predict_link_returns_confidence_and_explanation():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    model = LinkPredictionTrainer().fit(embeddings, dataset)
    predictor = LinkPredictor(repo)

    prediction = predictor.predict_link(ids[0], ids[1], embeddings, model)

    assert prediction is not None
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.explanation
    assert "confidence" in prediction.explanation.lower()


def test_predict_link_returns_none_for_a_node_without_an_embedding():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    model = LinkPredictionTrainer().fit(embeddings, dataset)
    predictor = LinkPredictor(repo)

    prediction = predictor.predict_link("does-not-exist", ids[1], embeddings, model)

    assert prediction is None


def test_top_candidates_excludes_already_connected_nodes():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    model = LinkPredictionTrainer().fit(embeddings, dataset)
    predictor = LinkPredictor(repo)

    candidates = predictor.top_candidates(ids[0], dataset, embeddings, model, top_k=10)

    candidate_targets = {c.target_id for c in candidates}
    assert ids[1] not in candidate_targets  # directly connected
    assert ids[-1] not in candidate_targets  # directly connected (ring closure)


def test_top_candidates_are_sorted_by_confidence_descending():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    model = LinkPredictionTrainer().fit(embeddings, dataset)
    predictor = LinkPredictor(repo)

    candidates = predictor.top_candidates(ids[0], dataset, embeddings, model, top_k=10)

    confidences = [c.confidence for c in candidates]
    assert confidences == sorted(confidences, reverse=True)


def test_top_candidates_respects_top_k():
    repo, dataset, ids = _seeded_dataset_and_repo()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    model = LinkPredictionTrainer().fit(embeddings, dataset)
    predictor = LinkPredictor(repo)

    candidates = predictor.top_candidates(ids[0], dataset, embeddings, model, top_k=2)

    assert len(candidates) <= 2
