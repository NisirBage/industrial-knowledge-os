from dataclasses import dataclass

from app.retrieval.ranking.features import RankingFeatures, extract_features


@dataclass
class _Candidate:
    bm25_score: float = 0.0
    vector_score: float = 0.0
    graph_score: float = 0.0
    fusion_score: float = 0.0
    text: str = ""


def test_extract_features_pulls_every_real_signal_from_the_candidate():
    candidate = _Candidate(bm25_score=3.5, vector_score=0.9, graph_score=0.15, fusion_score=0.032, text="hello world")

    features = extract_features(candidate, reranker_score=2.1)

    assert features.values == {
        "bm25_score": 3.5,
        "vector_score": 0.9,
        "graph_score": 0.15,
        "reranker_score": 2.1,
        "fusion_score": 0.032,
        "text_length": 11.0,
    }


def test_vector_returns_a_fixed_order_matching_feature_names():
    features = extract_features(_Candidate(bm25_score=1.0, vector_score=2.0, graph_score=3.0, fusion_score=4.0, text="ab"), reranker_score=5.0)

    assert features.vector() == [1.0, 2.0, 3.0, 5.0, 4.0, 2.0]
    assert features.vector() == [features.values[name] for name in RankingFeatures.feature_names()]


def test_vector_defaults_missing_named_features_to_zero():
    features = RankingFeatures({"bm25_score": 1.0})

    vector = features.vector()

    assert len(vector) == len(RankingFeatures.feature_names())
    assert vector[RankingFeatures.feature_names().index("bm25_score")] == 1.0
    assert vector[RankingFeatures.feature_names().index("vector_score")] == 0.0


def test_feature_names_is_stable_and_non_empty():
    names = RankingFeatures.feature_names()

    assert names == RankingFeatures.feature_names()
    assert len(names) > 0
