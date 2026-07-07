"""Domain-specific evaluators (M4 Component 6) — the "benchmark hooks" every
subsystem exposes. Each evaluator returns `list[NamedMetric]` uniformly, so
any of them can be logged, exported, or diffed across runs the same way
regardless of which subsystem produced them.
"""

from app.evaluation import metrics
from app.evaluation.models import NamedMetric


class RetrievalEvaluator:
    def evaluate(self, retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> list[NamedMetric]:
        retrieved_set = set(retrieved_ids)
        true_positive = len(retrieved_set & relevant_ids)
        false_positive = len(retrieved_set - relevant_ids)
        false_negative = len(relevant_ids - retrieved_set)

        precision_value = metrics.precision(true_positive, false_positive)
        recall_value = metrics.recall(true_positive, false_negative)
        relevance_scores = [1.0 if chunk_id in relevant_ids else 0.0 for chunk_id in retrieved_ids]

        return [
            NamedMetric("precision", precision_value),
            NamedMetric("recall", recall_value),
            NamedMetric("f1", metrics.f1_score(precision_value, recall_value)),
            NamedMetric(f"recall_at_{k}", metrics.recall_at_k(relevant_ids, retrieved_ids, k)),
            NamedMetric("ndcg", metrics.ndcg(relevance_scores)),
        ]


class EntityExtractionEvaluator:
    def evaluate(self, predicted: list[tuple[str, str]], ground_truth: list[tuple[str, str]]) -> list[NamedMetric]:
        predicted_set, truth_set = set(predicted), set(ground_truth)
        true_positive = len(predicted_set & truth_set)
        false_positive = len(predicted_set - truth_set)
        false_negative = len(truth_set - predicted_set)

        precision_value = metrics.precision(true_positive, false_positive)
        recall_value = metrics.recall(true_positive, false_negative)
        return [
            NamedMetric("precision", precision_value),
            NamedMetric("recall", recall_value),
            NamedMetric("f1", metrics.f1_score(precision_value, recall_value)),
        ]


class RelationExtractionEvaluator:
    def evaluate(self, predicted: list[tuple[str, str, str]], ground_truth: list[tuple[str, str, str]]) -> list[NamedMetric]:
        predicted_set, truth_set = set(predicted), set(ground_truth)
        true_positive = len(predicted_set & truth_set)
        false_positive = len(predicted_set - truth_set)
        false_negative = len(truth_set - predicted_set)

        precision_value = metrics.precision(true_positive, false_positive)
        recall_value = metrics.recall(true_positive, false_negative)
        return [
            NamedMetric("precision", precision_value),
            NamedMetric("recall", recall_value),
            NamedMetric("f1", metrics.f1_score(precision_value, recall_value)),
        ]


class RecommendationEvaluator:
    def evaluate(self, accepted_count: int, rejected_count: int) -> list[NamedMetric]:
        total = accepted_count + rejected_count
        acceptance_rate = round(accepted_count / total, 4) if total else 0.0
        return [
            NamedMetric("recommendation_acceptance_rate", acceptance_rate),
            NamedMetric("recommendation_feedback_volume", float(total)),
        ]


class PredictionEvaluator:
    def evaluate(self, predicted: list[float], actual: list[float]) -> list[NamedMetric]:
        mae = metrics.mean_absolute_error(predicted, actual)
        max_actual = max((abs(a) for a in actual), default=0.0)
        accuracy = round(max(0.0, 1 - mae / max_actual), 4) if max_actual else 0.0
        return [
            NamedMetric("mean_absolute_error", mae),
            NamedMetric("prediction_accuracy", accuracy),
        ]


class KnowledgeGraphEvaluator:
    def evaluate(self, node_count: int, edge_count: int) -> list[NamedMetric]:
        average_degree = round(edge_count / node_count, 4) if node_count else 0.0
        return [
            NamedMetric("node_count", float(node_count)),
            NamedMetric("edge_count", float(edge_count)),
            NamedMetric("average_degree", average_degree),
        ]


class FaithfulnessEvaluator:
    """Faithfulness/hallucination-rate proxy via lexical overlap between the
    narrated answer and its cited evidence — not a learned NLI faithfulness
    classifier. That's the natural upgrade path once ground-truth faithfulness
    labels exist; this heuristic is explainable and requires no training data.
    """

    def evaluate(self, answer_text: str, evidence_texts: list[str], cited_chunk_ids: list[str], known_chunk_ids: set[str]) -> list[NamedMetric]:
        answer_words = set(answer_text.lower().split())
        evidence_words: set[str] = set()
        for text in evidence_texts:
            evidence_words |= set(text.lower().split())

        overlap = len(answer_words & evidence_words)
        faithfulness = round(overlap / len(answer_words), 4) if answer_words else 0.0

        citation_accuracy = (
            round(len([c for c in cited_chunk_ids if c in known_chunk_ids]) / len(cited_chunk_ids), 4)
            if cited_chunk_ids
            else 1.0
        )

        return [
            NamedMetric("faithfulness", faithfulness),
            NamedMetric("hallucination_rate", round(1 - faithfulness, 4)),
            NamedMetric("citation_accuracy", citation_accuracy),
        ]


class CostEvaluator:
    """Token/cost estimate via a character-count approximation (~4 chars per
    token) since providers here don't return usage metadata — swap this for
    real usage-based accounting once the LLM provider surfaces it."""

    _CHARS_PER_TOKEN = 4

    def evaluate(self, prompt_text: str, response_text: str, rate_per_1k_tokens: float = 0.0) -> list[NamedMetric]:
        estimated_tokens = (len(prompt_text) + len(response_text)) / self._CHARS_PER_TOKEN
        estimated_cost = round(estimated_tokens / 1000 * rate_per_1k_tokens, 6)
        return [
            NamedMetric("estimated_tokens", round(estimated_tokens, 2)),
            NamedMetric("estimated_cost_usd", estimated_cost),
        ]


class LatencyEvaluator:
    """Uniform NamedMetric view over an existing latencies dict (e.g. the
    orchestrator's per-stage timings) — no new instrumentation, just a
    consistent export format."""

    def evaluate(self, latencies: dict[str, float]) -> list[NamedMetric]:
        return [NamedMetric(f"latency_{stage}", value) for stage, value in latencies.items()]
