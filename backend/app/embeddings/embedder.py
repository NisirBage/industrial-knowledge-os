"""Embedding generation.

Wraps a sentence-transformers model behind a stable interface so the model
can be swapped (e.g. BAAI/bge-m3 in production) without touching callers.
Falls back to a deterministic hashing embedder if the model can't be loaded
(no network access) so ingestion still works in offline demo environments —
retrieval quality degrades gracefully instead of the pipeline hard-failing.
"""

import hashlib
import logging
from typing import Protocol

logger = logging.getLogger(__name__)

_HASH_EMBEDDING_DIM = 384


class Embedder(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class HashingEmbedder:
    """Deterministic, dependency-free fallback embedder (not semantically meaningful)."""

    dimension = _HASH_EMBEDDING_DIM

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            vector[index] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]


def build_embedder(model_name: str) -> Embedder:
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:
        logger.warning(
            "Failed to load embedding model '%s' — falling back to HashingEmbedder. "
            "Retrieval quality will be degraded until a real model is available.",
            model_name,
        )
        return HashingEmbedder()
