"""Qdrant-backed repository for chunk embeddings (semantic search layer)."""

import logging
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.documents.chunking import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    score: float
    metadata: dict


class VectorRepository:
    def __init__(self, collection: str, client: QdrantClient):
        self._client = client
        self._collection = collection
        self._ensured = False

    def ensure_collection(self, vector_size: int) -> None:
        if self._ensured:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        self._ensured = True

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 8) -> list[ScoredChunk]:
        results = self._client.query_points(
            collection_name=self._collection, query=query_vector, limit=top_k
        ).points
        return [
            ScoredChunk(
                chunk_id=str(point.id),
                document_id=point.payload["document_id"],
                text=point.payload["text"],
                page_number=point.payload["page_number"],
                score=point.score,
                metadata={k: v for k, v in point.payload.items() if k not in {"document_id", "text", "page_number"}},
            )
            for point in results
        ]


def build_vector_repository(url: str, collection: str, local_path: str) -> VectorRepository:
    """Attempt a real Qdrant server connection; fall back to Qdrant's embedded
    local-file mode if the server is unreachable (e.g. no Docker locally), so
    the app still starts and semantic search still works, just single-process.
    """
    try:
        client = QdrantClient(url=url, timeout=2.0)
        client.get_collections()
        return VectorRepository(collection, client)
    except Exception:
        logger.warning(
            "Qdrant unreachable at %s — falling back to embedded local-file mode at %s.",
            url,
            local_path,
        )
        return VectorRepository(collection, QdrantClient(path=local_path))
