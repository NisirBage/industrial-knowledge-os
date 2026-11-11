from app.repositories.vector_repository import ScoredChunk
from app.retrieval.graph_expansion import GraphExpansionResult
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ranking.trainer import LTRTrainer


class _FakeEmbedder:
    dimension = 4

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _QueryAwareVectorRepository:
    """Returns different candidates depending on the query text, so
    different labeled queries produce genuinely different feature
    vectors/labels — required for LightGBM to have any real signal to
    learn from, rather than every query collapsing to identical data."""

    def __init__(self, hits_by_query: dict[str, list[ScoredChunk]]):
        self._hits_by_query = hits_by_query

    def search(self, query_vector, top_k=8):
        return self._current_hits[:top_k]

    def set_query(self, query: str):
        self._current_hits = self._hits_by_query.get(query, [])


class _QueryDispatchingRetrieverWrapper:
    """HybridRetriever's vector_repository.search() doesn't receive the raw
    query text (only its embedding) — so route through a small wrapper that
    remembers which query is in flight and updates the fake vector
    repository's active hit set before delegating, letting the fake return
    query-specific hits without changing HybridRetriever's real interface."""

    def __init__(self, retriever: HybridRetriever, vector_repository: _QueryAwareVectorRepository):
        self._retriever = retriever
        self._vector_repository = vector_repository

    def training_examples(self, query: str, top_k: int = 8):
        self._vector_repository.set_query(query)
        return self._retriever.training_examples(query, top_k=top_k)


class _FakeBm25Index:
    def search(self, query, top_k):
        return []


class _FakeGraphExpansion:
    def expand(self, query):
        return GraphExpansionResult(matched_entity_ids=[], boosted_document_ids=set(), graph_evidence=[])


class _ConstantReranker:
    def score(self, query, texts):
        return [0.0] * len(texts)


def _build_trainer() -> LTRTrainer:
    hits_by_query = {
        "query about pump": [
            ScoredChunk(chunk_id="c1", document_id="doc-pump", text="Pump P-101 maintenance history and specifications", page_number=1, score=0.9, metadata={}),
            ScoredChunk(chunk_id="c2", document_id="doc-valve", text="Valve V-202 unrelated content here", page_number=1, score=0.5, metadata={}),
        ],
        "query about valve": [
            ScoredChunk(chunk_id="c3", document_id="doc-valve", text="Valve V-202 maintenance history and specifications", page_number=1, score=0.9, metadata={}),
            ScoredChunk(chunk_id="c4", document_id="doc-pump", text="Pump P-101 unrelated content here", page_number=1, score=0.5, metadata={}),
        ],
        "query about motor": [
            ScoredChunk(chunk_id="c5", document_id="doc-motor", text="Motor M-303 maintenance history and specifications", page_number=1, score=0.9, metadata={}),
            ScoredChunk(chunk_id="c6", document_id="doc-pump", text="Pump P-101 unrelated content here", page_number=1, score=0.5, metadata={}),
        ],
    }
    vector_repository = _QueryAwareVectorRepository(hits_by_query)
    retriever = HybridRetriever(_FakeEmbedder(), vector_repository, _FakeBm25Index(), _FakeGraphExpansion(), _ConstantReranker())
    wrapper = _QueryDispatchingRetrieverWrapper(retriever, vector_repository)
    return LTRTrainer(wrapper), wrapper


def test_ltr_trainer_falls_back_to_untrained_ranker_with_fewer_than_two_labeled_queries():
    trainer, _ = _build_trainer()

    ranker = trainer.fit([("query about pump", {"doc-pump"})])

    assert ranker.is_trained is False


def test_ltr_trainer_falls_back_to_untrained_ranker_when_no_query_has_a_positive_label():
    trainer, _ = _build_trainer()

    ranker = trainer.fit([("query about pump", {"doc-nonexistent"}), ("query about valve", {"doc-also-nonexistent"})])

    assert ranker.is_trained is False


def test_ltr_trainer_produces_a_real_trained_ranker_from_labeled_queries():
    trainer, _ = _build_trainer()
    labeled_queries = [
        ("query about pump", {"doc-pump"}),
        ("query about valve", {"doc-valve"}),
        ("query about motor", {"doc-motor"}),
    ]

    ranker = trainer.fit(labeled_queries, n_estimators=5)

    assert ranker.is_trained is True
    assert hasattr(ranker.estimator, "feature_importances_")
    assert len(ranker.estimator.feature_importances_) > 0


def test_trained_ranker_produces_real_varying_scores_not_a_constant():
    trainer, _ = _build_trainer()
    labeled_queries = [
        ("query about pump", {"doc-pump"}),
        ("query about valve", {"doc-valve"}),
        ("query about motor", {"doc-motor"}),
    ]
    ranker = trainer.fit(labeled_queries, n_estimators=20)

    from app.retrieval.ranking.features import RankingFeatures

    scores = ranker.rank(
        "query",
        [
            RankingFeatures({"bm25_score": 0.0, "vector_score": 0.9, "graph_score": 0.0, "reranker_score": 0.0, "fusion_score": 0.02, "text_length": 50.0}),
            RankingFeatures({"bm25_score": 0.0, "vector_score": 0.1, "graph_score": 0.0, "reranker_score": 0.0, "fusion_score": 0.01, "text_length": 10.0}),
        ],
    )

    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
