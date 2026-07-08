"""GraphEmbeddingTrainer / GraphInferenceEngine (M9 Graph Intelligence API):
the two top-level orchestration classes the API layer talks to. Every
trained artifact (an `EmbeddingResult`, a `LinkPredictionModel`, a graph
`TrainedModel`) is a plain Python object with no unpicklable internals
(embedding vectors are plain `list[float]`, not live tensors) — so all of
them register into the *same* `SqlModelRegistry` M6 already built, "exactly
like the existing ML Laboratory... no parallel ML infrastructure" from the
M9 spec, literally true here rather than just similar in spirit.

Graph-embedding-based `GraphPredictionModel`s share `task_type` values with
M6's classical models (e.g. `failure_risk`) and are distinguished only by
`algorithm` (prefixed `graph_`) — so `GET /ml/models?task_type=failure_risk`
already lists both a rule/classical model and a graph model side by side,
without any new registry query needed for Component 9's rule-vs-graph
comparison.
"""

import hashlib
import json
from dataclasses import dataclass

from app.features.extractor import FeatureExtractor
from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.gnn import GATEmbedding, GCNEmbedding, GraphSAGEEmbedding
from app.graph_intelligence.embeddings.models import EmbeddingResult
from app.graph_intelligence.embeddings.node2vec import DeepWalkEmbedding, Node2VecEmbedding
from app.graph_intelligence.link_prediction import LinkPredictionModel, LinkPredictionTrainer
from app.graph_intelligence.node_classification import GraphNodeClassificationTrainer, GraphPredictionModel
from app.ml.registry.repository import ModelRegistryProtocol
from app.prediction.models import PredictionModel, PredictionTaskType
from app.repositories.graph_repository import GraphRepositoryProtocol

_EMBEDDING_MODELS = {
    "node2vec": Node2VecEmbedding,
    "deepwalk": DeepWalkEmbedding,
    "gcn": GCNEmbedding,
    "graphsage": GraphSAGEEmbedding,
    "gat": GATEmbedding,
}


def available_embedding_models() -> list[str]:
    return sorted(_EMBEDDING_MODELS)


def _hash_dataset(dataset: GraphDataset) -> str:
    payload = json.dumps(
        {"nodes": sorted(n.id for n in dataset.nodes), "edges": sorted(f"{e.source}->{e.target}:{e.relation_type}" for e in dataset.edges)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphEmbeddingTrainingResult:
    embedding_result: EmbeddingResult
    model_id: str


class GraphEmbeddingTrainer:
    def __init__(self, graph_repository: GraphRepositoryProtocol, model_registry: ModelRegistryProtocol):
        self._graph_repository = graph_repository
        self._model_registry = model_registry
        self._dataset_builder = GraphDatasetBuilder(graph_repository)

    async def train(self, model_name: str, **model_kwargs) -> GraphEmbeddingTrainingResult:
        model_cls = _EMBEDDING_MODELS.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unknown embedding model {model_name!r}. Available: {available_embedding_models()}")

        dataset = self._dataset_builder.build()
        model = model_cls(**model_kwargs)
        result = model.fit(dataset)

        record = await self._model_registry.register(
            estimator=result,
            task_type="graph_embedding",
            algorithm=model_name,
            training_data_hash=_hash_dataset(dataset),
            hyperparameters={k: str(v) for k, v in model_kwargs.items()},
            metrics={"node_count": float(len(dataset.nodes)), "edge_count": float(len(dataset.edges)), "training_seconds": result.training_seconds},
            feature_schema=[f"dim_{i}" for i in range(result.dimensions)],
        )
        return GraphEmbeddingTrainingResult(embedding_result=result, model_id=record.model_id)


class GraphInferenceEngine:
    """Loads a previously-trained embedding from the registry and runs
    link prediction / node classification / propagation against it — the
    single place the API layer goes for "do something with a trained graph
    model" rather than each route reconstructing training/loading logic."""

    def __init__(self, graph_repository: GraphRepositoryProtocol, model_registry: ModelRegistryProtocol):
        self._graph_repository = graph_repository
        self._model_registry = model_registry
        self._dataset_builder = GraphDatasetBuilder(graph_repository)

    async def load_latest_embedding(self) -> EmbeddingResult | None:
        record = await self._model_registry.get_latest("graph_embedding")
        if record is None:
            return None
        return await self._model_registry.load_estimator(record.model_id)

    async def train_link_predictor(self, embedding_result: EmbeddingResult) -> LinkPredictionModel:
        dataset = self._dataset_builder.build()
        model = LinkPredictionTrainer().fit(embedding_result, dataset)
        await self._model_registry.register(
            estimator=model,
            task_type="graph_link_prediction",
            algorithm="hadamard_logistic_regression",
            training_data_hash=_hash_dataset(dataset),
            hyperparameters={},
            metrics={},
            feature_schema=[],
        )
        return model

    async def train_node_classifier(
        self,
        task_type: PredictionTaskType,
        embedding_result: EmbeddingResult,
        feature_extractor: FeatureExtractor,
        rule_based_model: PredictionModel,
        equipment_canonical_names: list[str],
    ) -> GraphPredictionModel | None:
        dataset = self._dataset_builder.build()
        trained = await GraphNodeClassificationTrainer().fit_against_rule_baseline(
            task_type, embedding_result, dataset, feature_extractor, rule_based_model, equipment_canonical_names
        )
        if trained is None:
            return None

        await self._model_registry.register(
            estimator=trained,
            task_type=task_type.value,
            algorithm=f"graph_{trained.algorithm}",
            training_data_hash=_hash_dataset(dataset),
            hyperparameters=trained.hyperparameters,
            metrics=trained.metrics,
            feature_schema=trained.feature_names,
        )
        return GraphPredictionModel(task_type, trained, embedding_result, self._graph_repository)
