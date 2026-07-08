from app.features.extractor import RuleBasedFeatureExtractor
from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.embeddings.node2vec import Node2VecEmbedding
from app.graph_intelligence.node_classification import GraphNodeClassificationTrainer, GraphPredictionModel
from app.ml.explainability.explainer import Explainer
from app.ml.monitoring.monitor import ModelMonitor
from app.ontology.labels import EntityLabel, RelationType
from app.prediction.models import PredictionTaskType
from app.prediction.rule_models import build_rule_based_models
from app.repositories.graph_repository import InMemoryGraphRepository
from app.repositories.learning_repository import SqlLearningRepository
from app.timeline.engine import TimelineEngine


async def _learning_repo(tmp_path):
    db_path = tmp_path / "learning_test.db"
    repository = SqlLearningRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


def _seeded_repo_with_varied_risk():
    repo = InMemoryGraphRepository()
    equipment_ids = []
    for i in range(6):
        pump_id = repo.create_node(EntityLabel.PUMP, f"P-{i}")
        equipment_ids.append(pump_id)
        # Give some pumps more incidents than others so risk_score varies —
        # a constant target would make the "trained" model meaningless.
        for j in range(i % 3):
            incident_id = repo.create_node(EntityLabel.INCIDENT, f"incident-{i}-{j}")
            repo.create_relation(pump_id, incident_id, RelationType.FAILED_DUE_TO, confidence=0.5)
    # a chain so the graph is connected (embeddings need edges to be meaningful)
    for i in range(len(equipment_ids) - 1):
        repo.create_relation(equipment_ids[i], equipment_ids[i + 1], RelationType.CONNECTED_TO, confidence=0.5)
    return repo, ["P-0", "P-1", "P-2", "P-3", "P-4", "P-5"]


async def test_fit_against_rule_baseline_produces_a_trained_model(tmp_path):
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]

    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )

    assert trained is not None
    assert trained.task_type == PredictionTaskType.FAILURE_RISK.value
    assert len(trained.feature_names) == 8


async def test_fit_against_rule_baseline_returns_none_with_insufficient_data(tmp_path):
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-only")
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]

    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, ["P-only"]
    )

    assert trained is None


async def test_graph_prediction_model_implements_the_prediction_model_protocol(tmp_path):
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )

    graph_model = GraphPredictionModel(PredictionTaskType.FAILURE_RISK, trained, embeddings, repo)
    features = await feature_extractor.extract("P-0")

    result = graph_model.predict(features)

    assert result.value is not None
    assert 0.0 <= result.confidence <= 1.0
    assert "Graph embedding model" in result.explanation
    assert graph_model.task_type == PredictionTaskType.FAILURE_RISK


async def test_graph_prediction_model_handles_equipment_not_in_graph(tmp_path):
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )
    graph_model = GraphPredictionModel(PredictionTaskType.FAILURE_RISK, trained, embeddings, repo)

    features = await feature_extractor.extract("does-not-exist")
    result = graph_model.predict(features)

    assert result.value is None
    assert result.confidence == 0.0


async def test_trained_model_property_exposes_the_real_sklearn_estimator(tmp_path):
    """M9 Component 10 (ML Lab Integration): a graph model's underlying
    estimator must be reachable without poking a private attribute, so
    callers (the API layer, tests) can run the same Explainer M6's
    classical models use."""
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )
    graph_model = GraphPredictionModel(PredictionTaskType.FAILURE_RISK, trained, embeddings, repo)

    assert graph_model.trained_model is trained
    assert hasattr(graph_model.trained_model.estimator, "feature_importances_")


async def test_explainer_computes_real_feature_importance_for_a_graph_model(tmp_path):
    """Proves M6's Explainer (built for classical sklearn models) works
    unmodified on a graph-embedding model's estimator — "explainability ...
    exactly like the existing ML Laboratory," not a parallel mechanism."""
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )

    importance = Explainer().feature_importance(trained.estimator, trained.feature_names)

    assert len(importance) == 8
    assert {f.feature_name for f in importance} == {f"dim_{i}" for i in range(8)}
    assert all(imp.importance >= 0.0 for imp in importance)


async def test_model_monitor_computes_drift_on_real_graph_model_predictions(tmp_path):
    """Proves M6's ModelMonitor (PSI-based drift) works unmodified against
    a graph model's real prediction outputs — no graph-specific drift
    mechanism is needed since PSI only ever operates on plain float lists."""
    repo, equipment_names = _seeded_repo_with_varied_risk()
    dataset = GraphDatasetBuilder(repo).build()
    embeddings = Node2VecEmbedding(dimensions=8).fit(dataset)
    learning_repository = await _learning_repo(tmp_path)
    feature_extractor = RuleBasedFeatureExtractor(repo, TimelineEngine(repo), learning_repository)
    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
        PredictionTaskType.FAILURE_RISK, embeddings, dataset, feature_extractor, rule_model, equipment_names
    )
    graph_model = GraphPredictionModel(PredictionTaskType.FAILURE_RISK, trained, embeddings, repo)

    predictions = []
    for name in equipment_names:
        features = await feature_extractor.extract(name)
        predictions.append(graph_model.predict(features).value)

    monitor = ModelMonitor(learning_repository)
    psi = monitor.prediction_drift(predictions, predictions)

    assert psi == 0.0  # identical baseline/recent distributions -> no drift
    assert monitor.drift_severity(psi) == "none"
