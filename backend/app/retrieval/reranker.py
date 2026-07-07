"""Cross-encoder reranking over the merged (vector + BM25 + graph) candidate
set — never just the vector hits, since a candidate that only BM25 or the
graph surfaced deserves the same relevance judgment as one vector surfaced.

Defaults to a small cross-encoder for hackathon-speed CPU inference;
BAAI/bge-reranker-large (set via RERANKER_MODEL) is the production-quality
option once latency/hardware allow. Falls back to a neutral passthrough score
if the model can't be loaded, consistent with the embedder/LLM/GLiNER
offline-resilience pattern elsewhere in this codebase.
"""

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, text) for text in texts]
        return [float(s) for s in self._model.predict(pairs)]


class PassthroughReranker:
    """Neutral fallback: every candidate gets the same logit-scale score (0.0,
    i.e. sigmoid(0.0) = 0.5 confidence) so downstream merge ordering falls
    back to the fusion rank rather than a real cross-encoder judgment, while
    staying on the same logit scale the real CrossEncoderReranker uses."""

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)


def build_reranker(model_name: str) -> Reranker:
    try:
        return CrossEncoderReranker(model_name)
    except Exception:
        logger.warning(
            "Reranker model '%s' unavailable — falling back to PassthroughReranker "
            "(fusion rank order preserved, no cross-encoder re-scoring).",
            model_name,
        )
        return PassthroughReranker()
