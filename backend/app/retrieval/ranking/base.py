"""Ranker Protocol — the swap point `HybridRetriever` injects, mirroring the
existing `Reranker` Protocol pattern (`app/retrieval/reranker.py`) rather
than inventing a new extension style.
"""

from typing import Protocol

from app.retrieval.ranking.features import RankingFeatures


class Ranker(Protocol):
    def rank(self, query: str, feature_vectors: list[RankingFeatures]) -> list[float]:
        """Higher score = more relevant. Returns exactly one score per input
        feature vector, same order — caller sorts descending."""
        ...


class ReciprocalRankFusionRanker:
    """A real, named Reciprocal Rank Fusion ranker — no tunable weights, no
    training. `fusion_score` (already computed by `HybridRetriever._merge`
    as `sum(1 / (k + rank + 1))` across the BM25/vector rank lists, plus the
    flat graph-document boost) *is* RRF; this class exists so "rank by RRF
    alone, ignoring the cross-encoder" is a selectable, testable strategy
    rather than something only reachable by not having a cross-encoder at
    all.
    """

    def rank(self, query: str, feature_vectors: list[RankingFeatures]) -> list[float]:
        return [fv.values.get("fusion_score", 0.0) for fv in feature_vectors]


class NeuralRanker:
    """Extension point for a learned dense/late-interaction ranker (ColBERT,
    SPLADE, RankT5, MonoT5) — architecture-only, matching this project's
    established pattern for named-but-unimplemented extension points (DGL
    graph export, temporal graph edges in M9). Deliberately raises rather
    than silently falling back to a fake score: a caller that constructs
    this expecting real neural ranking should fail loudly, not get RRF
    scores mislabeled as a neural ranker's output."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def rank(self, query: str, feature_vectors: list[RankingFeatures]) -> list[float]:
        raise NotImplementedError(
            f"NeuralRanker({self._model_name!r}) is an architecture-only extension point — "
            "no ColBERT/SPLADE/RankT5/MonoT5 backend is implemented yet. "
            "Use ReciprocalRankFusionRanker or LightGBMRanker, or leave the ranker "
            "unset to fall back to the default cross-encoder-score ordering."
        )
