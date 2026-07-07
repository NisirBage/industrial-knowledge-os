"""In-process BM25 lexical index over ingested chunks.

Populated by the ingestion service alongside the vector store so every chunk
is searchable both lexically (here) and semantically (Qdrant) — the two
candidate lists are fused in hybrid.py. An in-process index is an MVP
simplification; production-scale would back this with an inverted index
service (e.g. OpenSearch/Elasticsearch BM25).
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.documents.chunking import Chunk


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    text: str
    page_number: int


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class Bm25Index:
    def __init__(self) -> None:
        self._chunks: list[IndexedChunk] = []
        self._bm25: BM25Okapi | None = None

    def add_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunks.append(
                IndexedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    page_number=chunk.page_number,
                )
            )
        self._rebuild()

    def _rebuild(self) -> None:
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[IndexedChunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [pair for pair in ranked[:top_k] if pair[1] > 0]
