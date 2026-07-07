from app.repositories.vector_repository import ScoredChunk
from app.retrieval.bm25_index import IndexedChunk
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionResult
from app.retrieval.hybrid import HybridRetriever


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
    def __init__(self, boosted_document_ids=frozenset(), graph_evidence=None):
        self._result = GraphExpansionResult(
            matched_entity_ids=[],
            boosted_document_ids=set(boosted_document_ids),
            graph_evidence=graph_evidence or [],
        )

    def expand(self, query):
        return self._result


class _ConstantReranker:
    def __init__(self, value=0.0):
        self._value = value

    def score(self, query, texts):
        return [self._value] * len(texts)


def test_retrieve_merges_vector_and_bm25_and_appends_graph_evidence():
    vector_hits = [ScoredChunk(chunk_id="v1", document_id="doc-A", text="vector hit", page_number=1, score=0.9, metadata={})]
    bm25_hits = [(IndexedChunk(chunk_id="b1", document_id="doc-B", text="bm25 hit", page_number=2), 3.5)]
    graph_evidence = [
        Evidence(document_id="", page_number=0, chunk_id="graph:1:2", text="P-101 CONNECTED_TO V-2201A", source="graph", confidence=0.55)
    ]

    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_repository=_FakeVectorRepository(vector_hits),
        bm25_index=_FakeBm25Index(bm25_hits),
        graph_expansion=_FakeGraphExpansion(graph_evidence=graph_evidence),
        reranker=_ConstantReranker(),
    )

    results = retriever.retrieve("how is P-101 connected", top_k=2)

    sources = [e.source for e in results]
    chunk_ids = {e.chunk_id for e in results}
    assert sources.count("fusion") == 2
    assert sources.count("graph") == 1
    assert {"v1", "b1", "graph:1:2"} == chunk_ids


def test_graph_boosted_document_ranks_above_equally_scored_candidate():
    vector_hits = [
        ScoredChunk(chunk_id="c1", document_id="doc-boosted", text="boosted doc chunk", page_number=1, score=0.5, metadata={}),
        ScoredChunk(chunk_id="c2", document_id="doc-plain", text="plain doc chunk", page_number=1, score=0.5, metadata={}),
    ]

    retriever = HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_repository=_FakeVectorRepository(vector_hits),
        bm25_index=_FakeBm25Index([]),
        graph_expansion=_FakeGraphExpansion(boosted_document_ids={"doc-boosted"}),
        reranker=_ConstantReranker(),  # equal reranker scores -> fusion order (with graph boost) decides
    )

    results = retriever.retrieve("query", top_k=2)
    fusion_results = [e for e in results if e.source == "fusion"]

    assert fusion_results[0].document_id == "doc-boosted"
    assert fusion_results[0].graph_score > 0.0
