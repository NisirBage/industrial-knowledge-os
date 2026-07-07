from app.evaluation.evaluators import (
    CostEvaluator,
    EntityExtractionEvaluator,
    FaithfulnessEvaluator,
    KnowledgeGraphEvaluator,
    LatencyEvaluator,
    PredictionEvaluator,
    RecommendationEvaluator,
    RelationExtractionEvaluator,
    RetrievalEvaluator,
)


def _metric(named_metrics, name):
    return next(m.value for m in named_metrics if m.name == name)


def test_retrieval_evaluator_perfect_retrieval():
    metrics = RetrievalEvaluator().evaluate(retrieved_ids=["a", "b"], relevant_ids={"a", "b"}, k=2)
    assert _metric(metrics, "precision") == 1.0
    assert _metric(metrics, "recall") == 1.0
    assert _metric(metrics, "f1") == 1.0


def test_retrieval_evaluator_partial_overlap():
    metrics = RetrievalEvaluator().evaluate(retrieved_ids=["a", "x"], relevant_ids={"a", "b"}, k=2)
    assert _metric(metrics, "precision") == 0.5
    assert _metric(metrics, "recall") == 0.5


def test_entity_extraction_evaluator_measures_label_and_name_match():
    predicted = [("Pump", "P-101"), ("Valve", "V-1")]
    truth = [("Pump", "P-101"), ("Valve", "V-2")]
    metrics = EntityExtractionEvaluator().evaluate(predicted, truth)
    assert _metric(metrics, "precision") == 0.5
    assert _metric(metrics, "recall") == 0.5


def test_relation_extraction_evaluator_measures_triple_match():
    predicted = [("P-101", "CONNECTED_TO", "V-1")]
    truth = [("P-101", "CONNECTED_TO", "V-1")]
    metrics = RelationExtractionEvaluator().evaluate(predicted, truth)
    assert _metric(metrics, "f1") == 1.0


def test_recommendation_evaluator_acceptance_rate():
    metrics = RecommendationEvaluator().evaluate(accepted_count=3, rejected_count=1)
    assert _metric(metrics, "recommendation_acceptance_rate") == 0.75
    assert _metric(metrics, "recommendation_feedback_volume") == 4.0


def test_recommendation_evaluator_zero_feedback():
    metrics = RecommendationEvaluator().evaluate(accepted_count=0, rejected_count=0)
    assert _metric(metrics, "recommendation_acceptance_rate") == 0.0


def test_prediction_evaluator_perfect_predictions():
    metrics = PredictionEvaluator().evaluate(predicted=[1.0, 2.0], actual=[1.0, 2.0])
    assert _metric(metrics, "mean_absolute_error") == 0.0
    assert _metric(metrics, "prediction_accuracy") == 1.0


def test_knowledge_graph_evaluator_average_degree():
    metrics = KnowledgeGraphEvaluator().evaluate(node_count=10, edge_count=25)
    assert _metric(metrics, "average_degree") == 2.5


def test_knowledge_graph_evaluator_empty_graph():
    metrics = KnowledgeGraphEvaluator().evaluate(node_count=0, edge_count=0)
    assert _metric(metrics, "average_degree") == 0.0


def test_faithfulness_evaluator_high_overlap_scores_high():
    answer = "Pump P-101 requires quarterly lubrication"
    evidence = ["Pump P-101 requires quarterly lubrication per the maintenance manual"]
    metrics = FaithfulnessEvaluator().evaluate(answer, evidence, cited_chunk_ids=["c1"], known_chunk_ids={"c1"})
    assert _metric(metrics, "faithfulness") > 0.8
    assert _metric(metrics, "citation_accuracy") == 1.0


def test_faithfulness_evaluator_flags_unknown_citations():
    metrics = FaithfulnessEvaluator().evaluate("answer text", ["evidence text"], cited_chunk_ids=["c1", "unknown"], known_chunk_ids={"c1"})
    assert _metric(metrics, "citation_accuracy") == 0.5


def test_cost_evaluator_estimates_tokens_and_cost():
    metrics = CostEvaluator().evaluate(prompt_text="a" * 400, response_text="b" * 400, rate_per_1k_tokens=1.0)
    assert _metric(metrics, "estimated_tokens") == 200.0
    assert _metric(metrics, "estimated_cost_usd") == 0.2


def test_latency_evaluator_wraps_dict_uniformly():
    metrics = LatencyEvaluator().evaluate({"planning": 0.1, "retrieval": 0.5})
    names = {m.name for m in metrics}
    assert names == {"latency_planning", "latency_retrieval"}
