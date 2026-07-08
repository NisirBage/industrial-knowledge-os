"""Link Prediction (M9 Component 5): predicts missing relationships (e.g.
Equipment-connected_to-Equipment, Incident-caused_by-Failure,
Recommendation-applies_to-Equipment) from graph embeddings. A real
scikit-learn `LogisticRegression` (the same library M6's ML Laboratory
already depends on — no new ML infrastructure) is trained on Hadamard-
product edge features: positive examples are real graph edges, negative
examples are randomly sampled non-edges, the same negative-sampling
discipline the GNN embedding models (gnn.py) already train with.

Every prediction carries confidence, the supporting neighborhood, embedding
similarity, and a full GraphExplanation — the M9 spec's explicit per-
prediction requirement.
"""

import random
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.embeddings.models import EmbeddingResult
from app.graph_intelligence.embeddings.operators import HadamardEdgeEmbedding
from app.graph_intelligence.explanation import GraphExplanationService
from app.graph_intelligence.similarity import GraphSimilarityService


@dataclass(frozen=True)
class LinkPrediction:
    source_id: str
    target_id: str
    confidence: float
    embedding_similarity: float
    supporting_neighborhood: list[str]
    explanation: str


class LinkPredictionModel:
    """A trained classifier over Hadamard edge features — `fit()` produces
    one of these, `predict_link()` consumes it. Kept as a real, inspectable
    object (not a bare sklearn estimator) so the Model Registry (M9
    Component 10) can serialize it like any other trained model."""

    def __init__(self, classifier: LogisticRegression, edge_embedding_model: HadamardEdgeEmbedding):
        self.classifier = classifier
        self.edge_embedding_model = edge_embedding_model


class LinkPredictionTrainer:
    def __init__(self, seed: int = 42):
        self._seed = seed
        self._edge_embedding_model = HadamardEdgeEmbedding()

    def fit(self, embedding_result: EmbeddingResult, dataset: GraphDataset) -> LinkPredictionModel:
        rng = random.Random(self._seed)
        node_ids = [n.id for n in dataset.nodes]
        existing_edges = {(e.source, e.target) for e in dataset.edges} | {(e.target, e.source) for e in dataset.edges}

        X: list[list[float]] = []
        y: list[int] = []
        for edge in dataset.edges:
            source_vector = embedding_result.vector_for(edge.source)
            target_vector = embedding_result.vector_for(edge.target)
            if source_vector is None or target_vector is None:
                continue
            X.append(self._edge_embedding_model.embed_edge(source_vector, target_vector))
            y.append(1)

        num_positive = len(X)
        attempts = 0
        while len(X) < num_positive * 2 and attempts < num_positive * 20 and len(node_ids) > 1:
            attempts += 1
            source, target = rng.sample(node_ids, 2)
            if (source, target) in existing_edges:
                continue
            source_vector = embedding_result.vector_for(source)
            target_vector = embedding_result.vector_for(target)
            if source_vector is None or target_vector is None:
                continue
            X.append(self._edge_embedding_model.embed_edge(source_vector, target_vector))
            y.append(0)

        classifier = LogisticRegression(max_iter=1000)
        if len(set(y)) >= 2:
            classifier.fit(X, y)
        return LinkPredictionModel(classifier=classifier, edge_embedding_model=self._edge_embedding_model)


class LinkPredictor:
    def __init__(self, graph_repository, explanation_service: GraphExplanationService | None = None):
        self._explanation_service = explanation_service or GraphExplanationService(graph_repository)
        self._similarity_service = GraphSimilarityService()

    def predict_link(
        self, source_id: str, target_id: str, embedding_result: EmbeddingResult, model: LinkPredictionModel
    ) -> LinkPrediction | None:
        source_vector = embedding_result.vector_for(source_id)
        target_vector = embedding_result.vector_for(target_id)
        if source_vector is None or target_vector is None:
            return None

        edge_features = model.edge_embedding_model.embed_edge(source_vector, target_vector)
        if hasattr(model.classifier, "classes_"):
            confidence = round(float(model.classifier.predict_proba([edge_features])[0][1]), 4)
        else:
            confidence = 0.5  # untrained fallback (too few edges to fit a real classifier)

        neighbors = self._similarity_service.nearest_neighbors(embedding_result, source_id, k=5)
        embedding_similarity = next((score for node_id, score in neighbors if node_id == target_id), 0.0)

        explanation = self._explanation_service.explain_node(source_id, embedding_result, confidence=confidence)
        supporting_neighborhood = [n["node_id"] for n in explanation.supporting_evidence]

        explanation_text = (
            f"Predicted link {source_id} -> {target_id} with confidence {round(confidence * 100, 1)}%, "
            f"embedding similarity {embedding_similarity}. {GraphExplanationService.to_text(explanation)}"
        )
        return LinkPrediction(
            source_id=source_id,
            target_id=target_id,
            confidence=confidence,
            embedding_similarity=embedding_similarity,
            supporting_neighborhood=supporting_neighborhood,
            explanation=explanation_text,
        )

    def top_candidates(
        self, source_id: str, dataset: GraphDataset, embedding_result: EmbeddingResult, model: LinkPredictionModel, top_k: int = 5
    ) -> list[LinkPrediction]:
        """Ranks every non-connected node as a candidate missing link for
        `source_id`, most-confident first — the "predict missing
        relationships" entry point the spec's Component 5 examples
        describe."""
        existing = {e.target for e in dataset.edges if e.source == source_id} | {e.source for e in dataset.edges if e.target == source_id}
        candidates = [n.id for n in dataset.nodes if n.id != source_id and n.id not in existing]

        predictions = []
        for candidate_id in candidates:
            prediction = self.predict_link(source_id, candidate_id, embedding_result, model)
            if prediction is not None:
                predictions.append(prediction)
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]
