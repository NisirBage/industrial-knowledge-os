"""Node Classification / Graph Intelligence Models (M9 Components 4 & 6).

Reuses `MLProblemType.GRAPH` — a value M6's `SklearnTrainer` already
declared but left with "no default algorithm... faking one would produce a
trained model that's really just noise" (see trainer.py). Supplying a real
algorithm factory for it here, rather than inventing a parallel training
path, is the literal extension point M6 built for this milestone.

Training target, honestly sourced: rather than fabricate labels no ground
truth exists for, each graph model is trained to reproduce the *existing
rule-based PredictionModel's* real output for the same task_type — the
model comparison this produces (rule baseline vs. graph-embedding model,
Component 9's benchmark) is genuine: two different computations (explicit
rule features vs. graph-structure embeddings) targeting the same real
quantity, not two models fit to invented numbers.

`GraphPredictionModel` implements the exact same `PredictionModel` protocol
(app.prediction.models) M4's rule-based models and M6's `MLPredictionModel`
do — a third, drop-in-compatible implementation, not a new prediction
interface.
"""

import statistics

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.features.extractor import FeatureExtractor
from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.models import EmbeddingResult
from app.graph_intelligence.explanation import GraphExplanationService
from app.ml.datasets.models import MLProblemType
from app.ml.training.models import TrainedModel
from app.ml.training.trainer import SklearnTrainer
from app.prediction.models import PredictionModel, PredictionResult, PredictionTaskType
from app.repositories.graph_repository import GraphRepositoryProtocol


def _default_graph_trainer() -> SklearnTrainer:
    return SklearnTrainer(algorithm_factories={MLProblemType.GRAPH: lambda: RandomForestRegressor(n_estimators=200, random_state=42)})


def _embedding_to_row(vector: list[float]) -> dict[str, float]:
    return {f"dim_{i}": value for i, value in enumerate(vector)}


class GraphNodeClassificationTrainer:
    def __init__(self, sklearn_trainer: SklearnTrainer | None = None):
        self._trainer = sklearn_trainer or _default_graph_trainer()

    async def fit_against_rule_baseline(
        self,
        task_type: PredictionTaskType,
        embedding_result: EmbeddingResult,
        dataset: GraphDataset,
        feature_extractor: FeatureExtractor,
        rule_based_model: PredictionModel,
        equipment_canonical_names: list[str],
    ) -> TrainedModel | None:
        """Returns None (rather than a degenerate model) if fewer than 2
        equipment ids resolve to both a graph embedding and a rule-based
        target — matching SklearnTrainer's own "don't fake a trained model"
        discipline for genuinely insufficient data."""
        canonical_to_node_id = {node.canonical_name: node.id for node in dataset.nodes}

        rows: list[dict[str, float]] = []
        targets: list[float] = []
        for equipment_id in equipment_canonical_names:
            node_id = canonical_to_node_id.get(equipment_id)
            if node_id is None:
                continue
            vector = embedding_result.vector_for(node_id)
            if vector is None:
                continue
            features = await feature_extractor.extract(equipment_id)
            target = rule_based_model.predict(features).value
            if target is None:
                continue
            rows.append(_embedding_to_row(vector))
            targets.append(target)

        if len(rows) < 2:
            return None

        X = pd.DataFrame(rows)
        y = pd.Series(targets)
        return self._trainer.train(X, y, MLProblemType.GRAPH, task_type.value)


class GraphPredictionModel:
    def __init__(
        self,
        task_type: PredictionTaskType,
        trained_model: TrainedModel,
        embedding_result: EmbeddingResult,
        graph_repository: GraphRepositoryProtocol,
        explanation_service: GraphExplanationService | None = None,
    ):
        self.task_type = task_type
        self._trained_model = trained_model
        self._embedding_result = embedding_result
        self._graph_repository = graph_repository
        self._explanation_service = explanation_service or GraphExplanationService(graph_repository)

    @property
    def trained_model(self) -> TrainedModel:
        """Public accessor for the underlying sklearn-backed TrainedModel —
        lets callers (e.g. the API layer) run the same Explainer used for
        M6's classical models against a graph model's real estimator,
        rather than reaching into a private attribute."""
        return self._trained_model

    def predict(self, features) -> PredictionResult:
        entity = self._graph_repository.find_entity(features.equipment_id)
        if entity is None:
            return PredictionResult(value=None, confidence=0.0, explanation="Equipment not found in the knowledge graph.")

        vector = self._embedding_result.vector_for(entity["id"])
        if vector is None:
            return PredictionResult(value=None, confidence=0.0, explanation="No graph embedding available for this equipment.")

        X = pd.DataFrame([_embedding_to_row(vector)])
        estimator = self._trained_model.estimator
        value = float(estimator.predict(X)[0])
        confidence = self._ensemble_confidence(estimator, X)

        explanation = self._explanation_service.explain_node(entity["id"], self._embedding_result, confidence=confidence)
        model_version = f"graph:{self._trained_model.algorithm}"
        explanation_text = (
            f"Graph embedding model ({model_version}) prediction: {round(value, 4)}. {GraphExplanationService.to_text(explanation)}"
        )
        return PredictionResult(value=round(value, 4), confidence=confidence, explanation=explanation_text, model_version=model_version)

    @staticmethod
    def _ensemble_confidence(estimator, X: pd.DataFrame) -> float:
        """Honest, real confidence proxy for a tree ensemble: lower variance
        across trees' individual predictions means the forest agrees more,
        which we treat as higher confidence — the same "real but simple
        heuristic, not a fabricated interval" standard M6's
        MLPredictionModel held its own uncertainty proxy to."""
        if not hasattr(estimator, "estimators_"):
            return 0.5
        tree_predictions = [tree.predict(X)[0] for tree in estimator.estimators_]
        if len(tree_predictions) < 2:
            return 0.5
        spread = statistics.pstdev(tree_predictions)
        return round(max(0.0, min(1.0, 1.0 - spread)), 4)
