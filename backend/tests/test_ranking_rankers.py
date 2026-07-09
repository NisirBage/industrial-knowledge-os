import pytest

from app.retrieval.ranking.base import NeuralRanker, ReciprocalRankFusionRanker
from app.retrieval.ranking.features import RankingFeatures
from app.retrieval.ranking.lightgbm_ranker import LightGBMRanker


def _features(fusion_score: float) -> RankingFeatures:
    return RankingFeatures({"fusion_score": fusion_score})


def test_rrf_ranker_returns_the_fusion_score_unchanged():
    ranker = ReciprocalRankFusionRanker()

    scores = ranker.rank("query", [_features(0.5), _features(0.1), _features(0.3)])

    assert scores == [0.5, 0.1, 0.3]


def test_rrf_ranker_handles_missing_fusion_score_as_zero():
    ranker = ReciprocalRankFusionRanker()

    scores = ranker.rank("query", [RankingFeatures({})])

    assert scores == [0.0]


def test_neural_ranker_raises_not_implemented_with_a_clear_message():
    ranker = NeuralRanker("colbert-v2")

    with pytest.raises(NotImplementedError, match="colbert-v2"):
        ranker.rank("query", [_features(0.5)])


def test_untrained_lightgbm_ranker_falls_back_to_fusion_score():
    ranker = LightGBMRanker()

    assert ranker.is_trained is False
    scores = ranker.rank("query", [_features(0.7), _features(0.2)])
    assert scores == [0.7, 0.2]


def test_untrained_lightgbm_ranker_handles_empty_input():
    ranker = LightGBMRanker()

    assert ranker.rank("query", []) == []


def test_trained_lightgbm_ranker_uses_the_real_estimator():
    import lightgbm as lgb

    X = [[1.0, 0.1], [0.1, 1.0], [0.5, 0.5], [0.9, 0.9]]
    y = [1, 0, 0, 1]
    estimator = lgb.LGBMRanker(n_estimators=5, verbosity=-1)
    estimator.fit(X, y, group=[4])
    ranker = LightGBMRanker(estimator)

    class _TwoFeatureVector:
        def __init__(self, a, b):
            self.values = {"a": a, "b": b}

        def vector(self):
            return [self.values["a"], self.values["b"]]

    scores = ranker.rank("query", [_TwoFeatureVector(1.0, 0.1), _TwoFeatureVector(0.1, 1.0)])

    assert ranker.is_trained is True
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
