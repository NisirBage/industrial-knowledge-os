"""Pure, dependency-free metric functions (M4 Component 6). Every evaluator
in evaluators.py is a thin domain-specific wrapper around these — the math
itself lives in exactly one place so retrieval, recommendation, and
prediction evaluation can't silently drift into inconsistent definitions of
"precision" or "recall."
"""

import math


def precision(true_positive: int, false_positive: int) -> float:
    denominator = true_positive + false_positive
    return round(true_positive / denominator, 4) if denominator else 0.0


def recall(true_positive: int, false_negative: int) -> float:
    denominator = true_positive + false_negative
    return round(true_positive / denominator, 4) if denominator else 0.0


def f1_score(precision_value: float, recall_value: float) -> float:
    denominator = precision_value + recall_value
    return round(2 * precision_value * recall_value / denominator, 4) if denominator else 0.0


def mrr(ranked_relevance: list[list[bool]]) -> float:
    """Mean Reciprocal Rank. Each inner list is one query's results in ranked
    order, True where that result is relevant."""
    if not ranked_relevance:
        return 0.0
    reciprocal_ranks = []
    for relevances in ranked_relevance:
        rank_score = 0.0
        for position, is_relevant in enumerate(relevances):
            if is_relevant:
                rank_score = 1.0 / (position + 1)
                break
        reciprocal_ranks.append(rank_score)
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return round(len(top_k & relevant_ids) / len(relevant_ids), 4)


def ndcg(relevance_scores: list[float], k: int | None = None) -> float:
    scores = relevance_scores[:k] if k is not None else relevance_scores
    dcg = sum(score / math.log2(i + 2) for i, score in enumerate(scores))
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_scores = ideal_scores[:k] if k is not None else ideal_scores
    idcg = sum(score / math.log2(i + 2) for i, score in enumerate(ideal_scores))
    return round(dcg / idcg, 4) if idcg else 0.0


def mean_absolute_error(predicted: list[float], actual: list[float]) -> float:
    if not predicted or len(predicted) != len(actual):
        return 0.0
    return round(sum(abs(p - a) for p, a in zip(predicted, actual, strict=True)) / len(predicted), 4)
