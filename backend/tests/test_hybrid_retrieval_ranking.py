from app.repositories.vector_repository import ScoredChunk
from app.retrieval.bm25_index import IndexedChunk
from app.retrieval.graph_expansion import GraphExpansionResult
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ranking.base import ReciprocalRankFusionRanker


class _FakeEmbedder:
    dimension = 4

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeVectorRepository:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_vector, top_k=8):
        return self._hits[:top_k]


class _FakeBm25Index:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


class _FakeGraphExpansion:
    def expand(self, query):
        return GraphExpansionResult(matched_entity_ids=[], boosted_document_ids=set(), graph_evidence=[])


class _InvertingReranker:
    """Scores the *first* candidate highest and every later one lower —
    used so the default (no-ranker) ordering is deterministic and easy to
    assert against, then contrasted with a ranker that reorders by a
    different signal (BM25 score) instead."""

    def score(self, query, texts):
        return [float(len(texts) - i) for i in range(len(texts))]


def test_default_no_ranker_orders_by_reranker_score():
    vector_hits = [
        ScoredChunk(chunk_id="v1", document_id="doc-A", text="alpha", page_number=1, score=0.1, metadata={}),
        ScoredChunk(chunk_id="v2", document_id="doc-B", text="beta", page_number=1, score=0.1, metadata={}),
    ]
    retriever = HybridRetriever(
        _FakeEmbedder(), _FakeVectorRepository(vector_hits), _FakeBm25Index([]), _FakeGraphExpansion(), _InvertingReranker()
    )

    results = retriever.retrieve("query", top_k=2)

    assert [e.chunk_id for e in results] == ["v1", "v2"]  # reranker gave v1 the higher score
    assert results[0].ltr_score == results[0].reranker_score


def test_rrf_ranker_reorders_by_fusion_score_instead_of_reranker_score():
    # v2 has a worse (lower) BM25 rank contribution than v1 in this setup,
    # but give v2 a higher raw BM25 score via bm25_hits ordering so its
    # fusion_score ends up higher than v1's — proving the RRF ranker
    # produces a *different* order than the reranker-only default above.
    vector_hits = [
        ScoredChunk(chunk_id="v1", document_id="doc-A", text="alpha", page_number=1, score=0.1, metadata={}),
        ScoredChunk(chunk_id="v2", document_id="doc-B", text="beta", page_number=1, score=0.1, metadata={}),
    ]
    bm25_hits = [
        (IndexedChunk(chunk_id="v2", document_id="doc-B", text="beta", page_number=1), 9.0),
    ]
    retriever = HybridRetriever(
        _FakeEmbedder(),
        _FakeVectorRepository(vector_hits),
        _FakeBm25Index(bm25_hits),
        _FakeGraphExpansion(),
        _InvertingReranker(),
        ranker=ReciprocalRankFusionRanker(),
    )

    results = retriever.retrieve("query", top_k=2)

    # v2 appears in both the vector and BM25 hit lists -> higher fusion_score
    # than v1 (vector-only) -> RRF ranker ranks it first, opposite of the
    # reranker-only default order asserted above for an equivalent setup.
    assert results[0].chunk_id == "v2"


def test_ranker_output_is_exposed_on_evidence_as_ltr_score_distinct_from_reranker_score():
    vector_hits = [ScoredChunk(chunk_id="v1", document_id="doc-A", text="alpha", page_number=1, score=0.1, metadata={})]
    retriever = HybridRetriever(
        _FakeEmbedder(),
        _FakeVectorRepository(vector_hits),
        _FakeBm25Index([]),
        _FakeGraphExpansion(),
        _InvertingReranker(),
        ranker=ReciprocalRankFusionRanker(),
    )

    results = retriever.retrieve("query", top_k=1)

    assert results[0].reranker_score == 1.0  # from _InvertingReranker with a single candidate
    assert results[0].ltr_score != results[0].reranker_score  # RRF's fusion_score, not the reranker score
    assert results[0].confidence > 0.0  # confidence still derives from reranker_score, unaffected by the ranker
