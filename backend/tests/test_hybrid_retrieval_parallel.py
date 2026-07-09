import time
from concurrent.futures import ThreadPoolExecutor

from app.repositories.vector_repository import ScoredChunk
from app.retrieval.bm25_index import IndexedChunk
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionResult
from app.retrieval.hybrid import HybridRetriever

_SLEEP_SECONDS = 0.05


class _SlowEmbedder:
    dimension = 4

    def embed_texts(self, texts):
        time.sleep(_SLEEP_SECONDS)
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _SlowVectorRepository:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_vector, top_k=8):
        time.sleep(_SLEEP_SECONDS)
        return self._hits[:top_k]


class _SlowBm25Index:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        time.sleep(_SLEEP_SECONDS)
        return self._hits[:top_k]


class _SlowGraphExpansion:
    def __init__(self, boosted_document_ids=frozenset(), graph_evidence=None):
        self._result = GraphExpansionResult(
            matched_entity_ids=[], boosted_document_ids=set(boosted_document_ids), graph_evidence=graph_evidence or []
        )

    def expand(self, query):
        time.sleep(_SLEEP_SECONDS)
        return self._result


class _ConstantReranker:
    def score(self, query, texts):
        return [0.0] * len(texts)


def _hits():
    vector_hits = [ScoredChunk(chunk_id="v1", document_id="doc-A", text="vector hit", page_number=1, score=0.9, metadata={})]
    bm25_hits = [(IndexedChunk(chunk_id="b1", document_id="doc-B", text="bm25 hit", page_number=2), 3.5)]
    return vector_hits, bm25_hits


def test_no_executor_runs_sequentially_and_takes_roughly_the_sum_of_the_three_delays():
    vector_hits, bm25_hits = _hits()
    retriever = HybridRetriever(
        _SlowEmbedder(), _SlowVectorRepository(vector_hits), _SlowBm25Index(bm25_hits), _SlowGraphExpansion(), _ConstantReranker()
    )

    start = time.perf_counter()
    retriever.retrieve("query", top_k=2)
    elapsed = time.perf_counter() - start

    # embed+vector (2x sleep, sequential within itself) + bm25 + expand = ~4x sleep
    assert elapsed >= _SLEEP_SECONDS * 3.5


def test_injected_executor_runs_the_three_independent_fetches_concurrently():
    vector_hits, bm25_hits = _hits()
    with ThreadPoolExecutor(max_workers=3) as executor:
        retriever = HybridRetriever(
            _SlowEmbedder(),
            _SlowVectorRepository(vector_hits),
            _SlowBm25Index(bm25_hits),
            _SlowGraphExpansion(),
            _ConstantReranker(),
            executor=executor,
        )

        start = time.perf_counter()
        retriever.retrieve("query", top_k=2)
        elapsed = time.perf_counter() - start

    # embed+vector (2x sleep, still sequential *within* that one branch) runs
    # concurrently with bm25 and expand -> bounded by the slowest branch
    # (~2x sleep), not the sum of all four blocking calls (~4x sleep).
    assert elapsed < _SLEEP_SECONDS * 3.5


def test_parallel_and_sequential_paths_produce_identical_evidence():
    vector_hits, bm25_hits = _hits()
    graph_evidence = [Evidence(document_id="", page_number=0, chunk_id="graph:1:2", text="triple", source="graph", confidence=0.55)]

    sequential = HybridRetriever(
        _SlowEmbedder(),
        _SlowVectorRepository(vector_hits),
        _SlowBm25Index(bm25_hits),
        _SlowGraphExpansion(graph_evidence=graph_evidence),
        _ConstantReranker(),
    )
    with ThreadPoolExecutor(max_workers=3) as executor:
        parallel = HybridRetriever(
            _SlowEmbedder(),
            _SlowVectorRepository(vector_hits),
            _SlowBm25Index(bm25_hits),
            _SlowGraphExpansion(graph_evidence=graph_evidence),
            _ConstantReranker(),
            executor=executor,
        )
        parallel_result = parallel.retrieve("query", top_k=2)

    sequential_result = sequential.retrieve("query", top_k=2)

    assert parallel_result == sequential_result


def test_executor_path_still_uses_the_embedding_cache():
    from app.caching.lru_cache import LRUTTLCache

    vector_hits, bm25_hits = _hits()
    embedding_cache = LRUTTLCache(max_size=16)
    with ThreadPoolExecutor(max_workers=3) as executor:
        retriever = HybridRetriever(
            _SlowEmbedder(),
            _SlowVectorRepository(vector_hits),
            _SlowBm25Index(bm25_hits),
            _SlowGraphExpansion(),
            _ConstantReranker(),
            embedding_cache=embedding_cache,
            executor=executor,
        )
        retriever.retrieve("query", top_k=2)
        retriever.retrieve("query", top_k=5)

    assert embedding_cache.stats.hits == 1
