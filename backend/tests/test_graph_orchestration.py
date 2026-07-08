from pathlib import Path

from app.features.extractor import RuleBasedFeatureExtractor
from app.graph_intelligence.orchestration import GraphEmbeddingTrainer, GraphInferenceEngine, available_embedding_models
from app.ml.registry.repository import SqlModelRegistry
from app.ontology.labels import EntityLabel, RelationType
from app.prediction.models import PredictionTaskType
from app.prediction.rule_models import build_rule_based_models
from app.repositories.graph_repository import InMemoryGraphRepository
from app.repositories.learning_repository import SqlLearningRepository
from app.timeline.engine import TimelineEngine


async def _registry(tmp_path):
    registry = SqlModelRegistry(f"sqlite+aiosqlite:///{tmp_path}/ml_registry.db", Path(tmp_path) / "models")
    await registry.init_models()
    return registry


async def _learning_repo(tmp_path):
    repository = SqlLearningRepository(f"sqlite+aiosqlite:///{tmp_path}/learning.db")
    await repository.init_models()
    return repository


def _seeded_repo_with_varied_risk():
    repo = InMemoryGraphRepository()
    equipment_names = []
    for i in range(6):
        pump_id = repo.create_node(EntityLabel.PUMP, f"P-{i}")
        equipment_names.append(f"P-{i}")
        for j in range(i % 3):
            incident_id = repo.create_node(EntityLabel.INCIDENT, f"incident-{i}-{j}")
            repo.create_relation(pump_id, incident_id, RelationType.FAILED_DUE_TO, confidence=0.5)
    ids = [repo.find_entity(name)["id"] for name in equipment_names]
    for i in range(len(ids) - 1):
        repo.create_relation(ids[i], ids[i + 1], RelationType.CONNECTED_TO, confidence=0.5)
    return repo, equipment_names


def test_available_embedding_models_lists_all_five():
    assert available_embedding_models() == ["deepwalk", "gat", "gcn", "graphsage", "node2vec"]


async def test_train_registers_an_embedding_in_the_model_registry(tmp_path):
    repo, _ = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    trainer = GraphEmbeddingTrainer(repo, registry)

    result = await trainer.train("node2vec", dimensions=8)

    assert len(result.embedding_result.embeddings) == repo.node_count()
    record = await registry.get(result.model_id)
    assert record is not None
    assert record.task_type == "graph_embedding"
    assert record.algorithm == "node2vec"


async def test_train_rejects_an_unknown_model_name(tmp_path):
    repo, _ = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    trainer = GraphEmbeddingTrainer(repo, registry)

    try:
        await trainer.train("not_a_real_model")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_model" in str(exc)


async def test_load_latest_embedding_returns_the_most_recently_trained_one(tmp_path):
    repo, _ = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    trainer = GraphEmbeddingTrainer(repo, registry)
    await trainer.train("node2vec", dimensions=8)

    inference_engine = GraphInferenceEngine(repo, registry)
    loaded = await inference_engine.load_latest_embedding()

    assert loaded is not None
    assert loaded.model_name == "node2vec"


async def test_load_latest_embedding_returns_none_when_nothing_trained(tmp_path):
    repo, _ = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    inference_engine = GraphInferenceEngine(repo, registry)

    assert await inference_engine.load_latest_embedding() is None


async def test_train_link_predictor_registers_a_model(tmp_path):
    repo, _ = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    trainer = GraphEmbeddingTrainer(repo, registry)
    embedding_result = (await trainer.train("node2vec", dimensions=8)).embedding_result

    inference_engine = GraphInferenceEngine(repo, registry)
    model = await inference_engine.train_link_predictor(embedding_result)

    assert model is not None
    history = await registry.list(task_type="graph_link_prediction")
    assert len(history) == 1


async def test_train_node_classifier_registers_alongside_the_rule_based_task_type(tmp_path):
    repo, equipment_names = _seeded_repo_with_varied_risk()
    registry = await _registry(tmp_path)
    trainer = GraphEmbeddingTrainer(repo, registry)
    embedding_result = (await trainer.train("node2vec", dimensions=8)).embedding_result

    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]

    inference_engine = GraphInferenceEngine(repo, registry)
    graph_model = await inference_engine.train_node_classifier(
        PredictionTaskType.FAILURE_RISK, embedding_result, feature_extractor, rule_model, equipment_names
    )

    assert graph_model is not None
    registered = await registry.list(task_type=PredictionTaskType.FAILURE_RISK.value)
    assert len(registered) == 1
    assert registered[0].algorithm.startswith("graph_")
