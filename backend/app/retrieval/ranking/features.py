"""Ranking feature extraction — every value here is computed from a signal
that genuinely exists in this deployment today.

The O1 spec names several additional features (ontology similarity, node
embedding similarity, OCR confidence, visual similarity, equipment match
score, maintenance history relevance, temporal recency, query intent as a
numeric input) that have no real backing data at the point this feature
vector is built: `_Candidate` (the fusion-stage record `HybridRetriever`
already builds) doesn't carry per-chunk entity ids, OCR confidence, or a
resolvable graph node id, and chunks carry no timestamp. Rather than fill
those in with fabricated zeros or constants that would silently never
change, they are simply absent from `_FEATURE_NAMES` — a feature that isn't
computed shows up as *missing*, not as a lie. See
docs/architecture/retrieval-optimization.md's roadmap for what each would
need (chunk-level entity ids, a query-to-graph-node resolver, per-chunk
timestamps) before it could be added honestly.
"""

from dataclasses import dataclass
from typing import Protocol

_FEATURE_NAMES = ["bm25_score", "vector_score", "graph_score", "reranker_score", "fusion_score", "text_length"]


class _CandidateLike(Protocol):
    bm25_score: float
    vector_score: float
    graph_score: float
    fusion_score: float
    text: str


@dataclass(frozen=True)
class RankingFeatures:
    values: dict[str, float]

    def vector(self) -> list[float]:
        """Fixed-order vector — same order every call, required for a
        trained model's `feature_importances_` to mean anything and for
        `predict()` to line up with what `fit()` saw."""
        return [self.values.get(name, 0.0) for name in _FEATURE_NAMES]

    @staticmethod
    def feature_names() -> list[str]:
        return list(_FEATURE_NAMES)


def extract_features(candidate: _CandidateLike, reranker_score: float) -> RankingFeatures:
    return RankingFeatures(
        {
            "bm25_score": candidate.bm25_score,
            "vector_score": candidate.vector_score,
            "graph_score": candidate.graph_score,
            "reranker_score": reranker_score,
            "fusion_score": candidate.fusion_score,
            "text_length": float(len(candidate.text)),
        }
    )
