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
from concurrent.futures import Executor
from dataclasses import dataclass

from app.caching.lru_cache import Cache, cache_key
from app.embeddings.embedder import Embedder
from app.repositories.vector_repository import VectorRepository
from app.retrieval.bm25_index import Bm25Index
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionResult, GraphExpansionStage
from app.retrieval.intent import classify_intent
from app.retrieval.ranking.base import Ranker
from app.retrieval.ranking.features import RankingFeatures, extract_features
from app.retrieval.reranker import Reranker

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
    """Optional caches (all default `None`, preserving exact prior behavior
    when omitted — see O1 Component 6). `result_cache` short-circuits the
    entire method for a repeated (query, top_k) pair. `embedding_cache` and
    `expansion_cache` are keyed on query text alone, so they're still reused
    across *different* `top_k` values for the same query — a case
    `result_cache` alone can't catch. `reranker_cache` is keyed per
    (query, chunk_id): the 12-wide rerank pool is independent of `top_k`
    (`_RERANK_POOL_SIZE`), so requests for the same query at different
    `top_k` values recompute an overlapping — often identical — pool; caching
    at this finer grain avoids re-scoring chunks the cross-encoder already
    judged for that exact query.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_repository: VectorRepository,
        bm25_index: Bm25Index,
        graph_expansion: GraphExpansionStage,
        reranker: Reranker,
        embedding_cache: Cache | None = None,
        expansion_cache: Cache | None = None,
        reranker_cache: Cache | None = None,
        result_cache: Cache | None = None,
        executor: Executor | None = None,
        ranker: Ranker | None = None,
    ):
        self._embedder = embedder
        self._vector_repository = vector_repository
        self._bm25_index = bm25_index
        self._graph_expansion = graph_expansion
        self._reranker = reranker
        self._embedding_cache = embedding_cache
        self._expansion_cache = expansion_cache
        self._reranker_cache = reranker_cache
        self._result_cache = result_cache
        # O1 Component 1: optional injected Ranker — when provided, replaces
        # "sort by cross-encoder score" as the final ordering step. `None`
        # (the default) preserves exact prior behavior.
        self._ranker = ranker
        # O1 Component 4: optional injected Executor — when provided, the
        # three independent blocking calls below (embed+vector-search,
        # BM25, graph expansion) run concurrently instead of sequentially.
        # `None` (the default) preserves the exact prior sequential
        # behavior, so every caller that never passes one is unaffected.
        self._executor = executor

    def set_ranker(self, ranker: Ranker | None) -> None:
        """Hot-swaps the ranker on the live instance — mirrors
        `PredictionEngine.register_model()`'s "train, then immediately serve
        with the new model" pattern (M4/M6/M9) rather than requiring an app
        restart to pick up a newly-trained ranker."""
        self._ranker = ranker

    def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
        if self._result_cache is not None:
            cached = self._result_cache.get(cache_key(query, top_k))
            if cached is not None:
                return cached

        expansion, pool, reranker_scores = self._scored_pool(query, top_k)

        if self._ranker is None:
            ltr_scores = reranker_scores
        else:
            feature_vectors = [extract_features(candidate, reranker_score) for candidate, reranker_score in zip(pool, reranker_scores, strict=True)]
            ltr_scores = self._ranker.rank(query, feature_vectors)

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
                ltr_score=ltr_score,
                source="fusion",
                confidence=round(_sigmoid(reranker_score), 2),
            )
            for candidate, reranker_score, ltr_score in zip(pool, reranker_scores, ltr_scores, strict=True)
        ]
        evidence.sort(key=lambda e: e.ltr_score, reverse=True)
        evidence = evidence[:top_k]

        result = evidence + expansion.graph_evidence
        if self._result_cache is not None:
            self._result_cache.set(cache_key(query, top_k), result)
        return result

    def training_examples(self, query: str, top_k: int = 8) -> list[tuple[str, "RankingFeatures"]]:
        """(document_id, RankingFeatures) for every candidate in the
        rerank pool for this query — the same pool `retrieve()` builds, just
        exposed before the final ranker/sort step. Used by `LTRTrainer`
        (O1 Component 1) to build labeled training data, and reusable by any
        future explainability tooling that needs "what features did this
        query's candidates actually have" without duplicating the
        fetch/merge/rerank pipeline."""
        _expansion, pool, reranker_scores = self._scored_pool(query, top_k)
        return [
            (candidate.document_id, extract_features(candidate, reranker_score))
            for candidate, reranker_score in zip(pool, reranker_scores, strict=True)
        ]

    def _scored_pool(self, query: str, top_k: int) -> tuple[GraphExpansionResult, list["_Candidate"], list[float]]:
        classify_intent(query)  # available for future metadata-filter routing
        candidate_pool = max(top_k * 3, _RERANK_POOL_SIZE)

        expansion, vector_hits, bm25_hits = self._fetch_candidates(query, candidate_pool)

        candidates = self._merge(vector_hits, bm25_hits, expansion.boosted_document_ids)
        candidates = _metadata_filter(candidates)

        pool = sorted(candidates.values(), key=lambda c: c.fusion_score, reverse=True)[:_RERANK_POOL_SIZE]
        reranker_scores = self._cached_rerank(query, pool)
        return expansion, pool, reranker_scores

    def _fetch_candidates(self, query: str, candidate_pool: int):
        """Three independent, blocking fetches — graph expansion, BM25
        search, and embed-then-vector-search — with no data dependency
        between them (only the vector search step itself depends on its own
        embedding). Sequential when `self._executor` is `None` (identical to
        the pre-O1 behavior); concurrent via the injected `Executor`
        otherwise, so overall latency is bounded by the slowest of the three
        rather than their sum."""
        if self._executor is None:
            expansion = self._cached_expand(query)
            query_vector = self._cached_embed(query)
            vector_hits = self._vector_repository.search(query_vector, top_k=candidate_pool)
            bm25_hits = self._bm25_index.search(query, top_k=candidate_pool)
            return expansion, vector_hits, bm25_hits

        def _fetch_vector_hits():
            query_vector = self._cached_embed(query)
            return self._vector_repository.search(query_vector, top_k=candidate_pool)

        expansion_future = self._executor.submit(self._cached_expand, query)
        vector_future = self._executor.submit(_fetch_vector_hits)
        bm25_future = self._executor.submit(self._bm25_index.search, query, candidate_pool)

        return expansion_future.result(), vector_future.result(), bm25_future.result()

    def _cached_embed(self, query: str) -> list[float]:
        if self._embedding_cache is None:
            return self._embedder.embed_texts([query])[0]
        cached = self._embedding_cache.get(query)
        if cached is not None:
            return cached
        vector = self._embedder.embed_texts([query])[0]
        self._embedding_cache.set(query, vector)
        return vector

    def _cached_expand(self, query: str) -> GraphExpansionResult:
        if self._expansion_cache is None:
            return self._graph_expansion.expand(query)
        cached = self._expansion_cache.get(query)
        if cached is not None:
            return cached
        result = self._graph_expansion.expand(query)
        self._expansion_cache.set(query, result)
        return result

    def _cached_rerank(self, query: str, pool: list["_Candidate"]) -> list[float]:
        if self._reranker_cache is None or not pool:
            return self._reranker.score(query, [c.text for c in pool])

        scores: list[float | None] = []
        miss_indices: list[int] = []
        for candidate in pool:
            score = self._reranker_cache.get(cache_key(query, candidate.chunk_id))
            scores.append(score)
            if score is None:
                miss_indices.append(len(scores) - 1)

        if miss_indices:
            miss_scores = self._reranker.score(query, [pool[i].text for i in miss_indices])
            for index, score in zip(miss_indices, miss_scores, strict=True):
                scores[index] = score
                self._reranker_cache.set(cache_key(query, pool[index].chunk_id), score)

        return scores  # type: ignore[return-value]  -- every slot filled above

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
