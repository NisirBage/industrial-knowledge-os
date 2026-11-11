from app.caching.lru_cache import LRUTTLCache
from app.repositories.vector_repository import ScoredChunk
from app.retrieval.bm25_index import IndexedChunk
from app.retrieval.graph_expansion import GraphExpansionResult
from app.retrieval.hybrid import HybridRetriever


class _FakeEmbedder:
    dimension = 4

    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    def search(self, query_vector, top_k=8):
        self.calls += 1
        return self._hits[:top_k]


class _FakeBm25Index:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class _FakeGraphExpansion:
    def __init__(self, boosted_document_ids=frozenset(), graph_evidence=None):
        self._result = GraphExpansionResult(
            matched_entity_ids=[], boosted_document_ids=set(boosted_document_ids), graph_evidence=graph_evidence or []
        )
        self.calls = 0

    def expand(self, query):
        self.calls += 1
        return self._result


class _CountingReranker:
    def __init__(self, value=0.0):
        self._value = value
        self.calls = 0
        self.scored_texts: list[str] = []

    def score(self, query, texts):
        self.calls += 1
        self.scored_texts.extend(texts)
        return [self._value] * len(texts)


def _retriever(embedder, vector_repository, bm25_index, graph_expansion, reranker, **caches):
    return HybridRetriever(
        embedder=embedder,
        vector_repository=vector_repository,
        bm25_index=bm25_index,
        graph_expansion=graph_expansion,
        reranker=reranker,
        **caches,
    )


def _hits():
    vector_hits = [ScoredChunk(chunk_id="v1", document_id="doc-A", text="vector hit", page_number=1, score=0.9, metadata={})]
    bm25_hits = [(IndexedChunk(chunk_id="b1", document_id="doc-B", text="bm25 hit", page_number=2), 3.5)]
    return vector_hits, bm25_hits


def test_result_cache_hit_skips_every_downstream_call_on_the_second_identical_request():
    vector_hits, bm25_hits = _hits()
    embedder = _FakeEmbedder()
    vector_repository = _FakeVectorRepository(vector_hits)
    graph_expansion = _FakeGraphExpansion()
    reranker = _CountingReranker()
    result_cache = LRUTTLCache(max_size=16)

    retriever = _retriever(embedder, vector_repository, _FakeBm25Index(bm25_hits), graph_expansion, reranker, result_cache=result_cache)

    first = retriever.retrieve("how is P-101 connected", top_k=2)
    second = retriever.retrieve("how is P-101 connected", top_k=2)

    assert first == second
    assert embedder.calls == 1
    assert vector_repository.calls == 1
    assert graph_expansion.calls == 1
    assert reranker.calls == 1
    assert result_cache.stats.hits == 1
    assert result_cache.stats.misses == 1


def test_result_cache_is_keyed_by_top_k_not_just_query():
    vector_hits, bm25_hits = _hits()
    result_cache = LRUTTLCache(max_size=16)
    retriever = _retriever(
        _FakeEmbedder(), _FakeVectorRepository(vector_hits), _FakeBm25Index(bm25_hits), _FakeGraphExpansion(), _CountingReranker(),
        result_cache=result_cache,
    )

    retriever.retrieve("query", top_k=2)
    retriever.retrieve("query", top_k=5)  # different top_k -> not a cache hit

    assert result_cache.stats.misses == 2
    assert result_cache.stats.hits == 0


def test_embedding_cache_is_reused_across_different_top_k_for_the_same_query():
    vector_hits, bm25_hits = _hits()
    embedder = _FakeEmbedder()
    embedding_cache = LRUTTLCache(max_size=16)
    retriever = _retriever(
        embedder, _FakeVectorRepository(vector_hits), _FakeBm25Index(bm25_hits), _FakeGraphExpansion(), _CountingReranker(),
        embedding_cache=embedding_cache,
    )

    retriever.retrieve("query", top_k=2)
    retriever.retrieve("query", top_k=5)  # no result_cache, but embedding should still be reused

    assert embedder.calls == 1
    assert embedding_cache.stats.hits == 1


def test_expansion_cache_is_reused_across_different_top_k_for_the_same_query():
    vector_hits, bm25_hits = _hits()
    graph_expansion = _FakeGraphExpansion()
    expansion_cache = LRUTTLCache(max_size=16)
    retriever = _retriever(
        _FakeEmbedder(), _FakeVectorRepository(vector_hits), _FakeBm25Index(bm25_hits), graph_expansion, _CountingReranker(),
        expansion_cache=expansion_cache,
    )

    retriever.retrieve("query", top_k=2)
    retriever.retrieve("query", top_k=5)

    assert graph_expansion.calls == 1
    assert expansion_cache.stats.hits == 1


def test_reranker_cache_avoids_rescoring_the_same_chunk_for_the_same_query_at_different_top_k():
    vector_hits, bm25_hits = _hits()
    reranker = _CountingReranker()
    reranker_cache = LRUTTLCache(max_size=16)
    retriever = _retriever(
        _FakeEmbedder(), _FakeVectorRepository(vector_hits), _FakeBm25Index(bm25_hits), _FakeGraphExpansion(), reranker,
        reranker_cache=reranker_cache,
    )

    retriever.retrieve("query", top_k=1)
    retriever.retrieve("query", top_k=2)  # same 2-item pool, already fully cached

    assert reranker.calls == 1  # second call finds every candidate already cached, never calls the real reranker
    assert reranker_cache.stats.hits >= 1


def test_reranker_cache_is_keyed_per_query_not_shared_across_different_queries():
    """A cache hit requires the same (query, chunk_id) pair — reusing a
    chunk's rerank score for a *different* query would be wrong (the
    cross-encoder judges query-chunk relevance, not chunk quality alone), so
    a different query must always be a real, uncached rerank call."""
    reranker = _CountingReranker()
    reranker_cache = LRUTTLCache(max_size=16)
    vector_hits = [
        ScoredChunk(chunk_id="v1", document_id="doc-A", text="hit one", page_number=1, score=0.9, metadata={}),
        ScoredChunk(chunk_id="v2", document_id="doc-A", text="hit two", page_number=1, score=0.8, metadata={}),
    ]
    retriever = _retriever(
        _FakeEmbedder(), _FakeVectorRepository(vector_hits), _FakeBm25Index([]), _FakeGraphExpansion(), reranker,
        reranker_cache=reranker_cache,
    )
    retriever.retrieve("query", top_k=2)
    reranker.scored_texts.clear()

    retriever.retrieve("a different query", top_k=2)

    assert reranker.calls == 2  # one real rerank call per distinct query
    assert set(reranker.scored_texts) == {"hit one", "hit two"}  # the different query rescored both, none reused


def test_no_caches_provided_preserves_prior_behavior_recomputing_every_call():
    vector_hits, bm25_hits = _hits()
    embedder = _FakeEmbedder()
    retriever = _retriever(embedder, _FakeVectorRepository(vector_hits), _FakeBm25Index(bm25_hits), _FakeGraphExpansion(), _CountingReranker())

    retriever.retrieve("query", top_k=2)
    retriever.retrieve("query", top_k=2)

    assert embedder.calls == 2
