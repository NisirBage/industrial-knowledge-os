"""Graph-aware hybrid retrieval.

Question -> Intent Classification -> Metadata Filter -> Graph Expansion ->
Vector Search -> BM25 -> Merge (reciprocal rank fusion + graph boost) ->
Cross-Encoder Rerank -> Evidence list (Context Builder).

Never retrieves by embeddings alone, and the graph is consulted *before*
vector/BM25 search finishes (not just as a post-hoc boost): graph_expansion
also contributes its own synthesized relation-triple evidence directly into
the reranked set.
"""

import math
from dataclasses import dataclass

from app.embeddings.embedder import Embedder
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionStage
from app.retrieval.intent import classify_intent
from app.retrieval.reranker import Reranker
from app.repositories.vector_repository import VectorRepository
from app.retrieval.bm25_index import Bm25Index

_RRF_K = 60
_GRAPH_DOCUMENT_BOOST = 0.15
_RERANK_POOL_SIZE = 12


@dataclass
class _Candidate:
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    fusion_score: float = 0.0


def _metadata_filter(candidates: list[_Candidate]) -> list[_Candidate]:
    """Extension point: once ingestion captures richer per-chunk metadata
    (document type, department, date), filter/boost candidates by
    QueryIntent here before graph expansion and search. Currently a
    passthrough — no metadata beyond content_type exists yet to filter on."""
    return candidates


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_repository: VectorRepository,
        bm25_index: Bm25Index,
        graph_expansion: GraphExpansionStage,
        reranker: Reranker,
    ):
        self._embedder = embedder
        self._vector_repository = vector_repository
        self._bm25_index = bm25_index
        self._graph_expansion = graph_expansion
        self._reranker = reranker

    def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
        classify_intent(query)  # available for future metadata-filter routing
        candidate_pool = max(top_k * 3, _RERANK_POOL_SIZE)

        expansion = self._graph_expansion.expand(query)

        query_vector = self._embedder.embed_texts([query])[0]
        vector_hits = self._vector_repository.search(query_vector, top_k=candidate_pool)
        bm25_hits = self._bm25_index.search(query, top_k=candidate_pool)

        candidates = self._merge(vector_hits, bm25_hits, expansion.boosted_document_ids)
        candidates = _metadata_filter(candidates)

        pool = sorted(candidates.values(), key=lambda c: c.fusion_score, reverse=True)[:_RERANK_POOL_SIZE]
        reranker_scores = self._reranker.score(query, [c.text for c in pool])

        evidence = [
            Evidence(
                document_id=candidate.document_id,
                page_number=candidate.page_number,
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                vector_score=candidate.vector_score,
                bm25_score=candidate.bm25_score,
                graph_score=candidate.graph_score,
                reranker_score=reranker_score,
                source="fusion",
                confidence=round(_sigmoid(reranker_score), 2),
            )
            for candidate, reranker_score in zip(pool, reranker_scores, strict=True)
        ]
        evidence.sort(key=lambda e: e.reranker_score, reverse=True)
        evidence = evidence[:top_k]

        return evidence + expansion.graph_evidence

    def _merge(self, vector_hits, bm25_hits, boosted_document_ids: set[str]) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}

        for rank, hit in enumerate(vector_hits):
            candidate = candidates.setdefault(
                hit.chunk_id,
                _Candidate(chunk_id=hit.chunk_id, document_id=hit.document_id, text=hit.text, page_number=hit.page_number),
            )
            candidate.vector_score = hit.score
            candidate.fusion_score += 1.0 / (_RRF_K + rank + 1)

        for rank, (indexed_chunk, bm25_score) in enumerate(bm25_hits):
            candidate = candidates.setdefault(
                indexed_chunk.chunk_id,
                _Candidate(
                    chunk_id=indexed_chunk.chunk_id,
                    document_id=indexed_chunk.document_id,
                    text=indexed_chunk.text,
                    page_number=indexed_chunk.page_number,
                ),
            )
            candidate.bm25_score = bm25_score
            candidate.fusion_score += 1.0 / (_RRF_K + rank + 1)

        for candidate in candidates.values():
            if candidate.document_id in boosted_document_ids:
                candidate.graph_score = _GRAPH_DOCUMENT_BOOST
                candidate.fusion_score += _GRAPH_DOCUMENT_BOOST

        return candidates
