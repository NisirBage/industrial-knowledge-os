from app.evaluation import metrics


def test_precision_basic():
    assert metrics.precision(true_positive=8, false_positive=2) == 0.8


def test_precision_zero_denominator():
    assert metrics.precision(0, 0) == 0.0


def test_recall_basic():
    assert metrics.recall(true_positive=8, false_negative=2) == 0.8


def test_f1_score_balances_precision_and_recall():
    assert metrics.f1_score(precision_value=1.0, recall_value=0.0) == 0.0
    assert metrics.f1_score(precision_value=0.8, recall_value=0.8) == 0.8


def test_mrr_rewards_earlier_relevant_results():
    assert metrics.mrr([[True, False, False]]) == 1.0
    assert metrics.mrr([[False, True, False]]) == 0.5
    assert metrics.mrr([[False, False, False]]) == 0.0


def test_mrr_empty_input():
    assert metrics.mrr([]) == 0.0


def test_recall_at_k_only_considers_top_k():
    relevant = {"a", "b", "c"}
    retrieved = ["a", "x", "y", "b", "c"]
    assert metrics.recall_at_k(relevant, retrieved, k=2) == round(1 / 3, 4)
    assert metrics.recall_at_k(relevant, retrieved, k=5) == 1.0


def test_recall_at_k_empty_relevant_set():
    assert metrics.recall_at_k(set(), ["a", "b"], k=2) == 0.0


def test_ndcg_perfect_ranking_scores_one():
    assert metrics.ndcg([1.0, 1.0, 0.0]) == 1.0


def test_ndcg_zero_relevance_scores_zero():
    assert metrics.ndcg([0.0, 0.0]) == 0.0


def test_mean_absolute_error_basic():
    assert metrics.mean_absolute_error([1.0, 2.0], [1.0, 4.0]) == 1.0


def test_mean_absolute_error_mismatched_lengths_returns_zero():
    assert metrics.mean_absolute_error([1.0], [1.0, 2.0]) == 0.0
