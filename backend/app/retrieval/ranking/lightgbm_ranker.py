"""LightGBM LambdaMART ranker — the trainable `Ranker` implementation.

`LightGBMRanker` wraps a plain `lightgbm.LGBMRanker`, which satisfies the
same `SklearnEstimator` duck-type (`fit`/`predict`/`feature_importances_`)
every other ML Lab model already does — so it registers into and is
explained by the *same* `SqlModelRegistry`/`Explainer` M6 built, per this
project's "no parallel ML infrastructure" standard (see M9's Graph
Intelligence for the same pattern applied to graph embeddings).
"""

from app.retrieval.ranking.features import RankingFeatures


class LightGBMRanker:
    """Wraps a trained `lightgbm.LGBMRanker`. `rank()` falls back to the
    `fusion_score` feature (the same Reciprocal Rank Fusion score
    `ReciprocalRankFusionRanker` uses) when the model hasn't been trained
    yet — the same graceful-fallback discipline the embedder/reranker/LLM
    provider already follow elsewhere in this codebase, rather than raising
    or returning a meaningless constant."""

    def __init__(self, estimator=None):
        self._estimator = estimator

    @property
    def estimator(self):
        return self._estimator

    @property
    def is_trained(self) -> bool:
        return self._estimator is not None

    def rank(self, query: str, feature_vectors: list[RankingFeatures]) -> list[float]:
        if not feature_vectors:
            return []
        if self._estimator is None:
            return [fv.values.get("fusion_score", 0.0) for fv in feature_vectors]
        X = [fv.vector() for fv in feature_vectors]
        return [float(s) for s in self._estimator.predict(X)]
