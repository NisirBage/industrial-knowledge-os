from app.research.benchmarks.graph_runners import (
    GraphEmbeddingBenchmark,
    LinkPredictionBenchmark,
    NodeClassificationBenchmark,
)


async def test_graph_embedding_benchmark_runs_for_every_model_name():
    for model_name in ["node2vec", "deepwalk", "gcn", "graphsage", "gat"]:
        report = await GraphEmbeddingBenchmark(model_name, num_nodes=12).run()
        assert report.benchmark_name == f"graph_embedding_{model_name}"
        metric_names = {m.name for m in report.metrics}
        assert metric_names == {
            "training_seconds", "peak_memory_bytes", "mean_connected_similarity",
            "mean_random_similarity", "embedding_quality_margin",
        }


async def test_graph_embedding_benchmark_rejects_an_unknown_model_name():
    try:
        GraphEmbeddingBenchmark("not_a_real_model")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "not_a_real_model" in str(exc)


async def test_graph_embedding_benchmark_reports_positive_training_time():
    report = await GraphEmbeddingBenchmark("node2vec", num_nodes=12).run()

    training_seconds = next(m.value for m in report.metrics if m.name == "training_seconds")
    assert training_seconds >= 0.0


async def test_link_prediction_benchmark_reports_mrr_and_recall():
    report = await LinkPredictionBenchmark(num_nodes=16).run()

    metric_names = {m.name for m in report.metrics}
    assert metric_names == {"mrr", "recall_at_5", "held_out_edges"}
    mrr_value = next(m.value for m in report.metrics if m.name == "mrr")
    assert 0.0 <= mrr_value <= 1.0


async def test_link_prediction_benchmark_evaluates_against_a_real_holdout_not_already_connected_nodes():
    """Regression test: an earlier version evaluated held-out edges without
    removing them from the training graph first, so the true target was
    always excluded from ranking by top_candidates()'s own "already
    connected" filter — guaranteeing MRR/Recall@5 of exactly 0.0 regardless
    of model quality. A real (if imperfect) benchmark should not be
    hard-wired to zero."""
    report = await LinkPredictionBenchmark(num_nodes=20).run()

    mrr_value = next(m.value for m in report.metrics if m.name == "mrr")
    recall_value = next(m.value for m in report.metrics if m.name == "recall_at_5")
    assert mrr_value > 0.0 or recall_value > 0.0


async def test_node_classification_benchmark_reports_both_maes():
    report = await NodeClassificationBenchmark(num_nodes=16).run()

    metric_names = {m.name for m in report.metrics}
    assert metric_names == {"rule_baseline_mae", "graph_model_mae", "graph_model_wins"}
    rule_mae = next(m.value for m in report.metrics if m.name == "rule_baseline_mae")
    graph_mae = next(m.value for m in report.metrics if m.name == "graph_model_mae")
    assert rule_mae >= 0.0
    assert graph_mae >= 0.0


async def test_node_classification_benchmark_wins_flag_matches_the_maes():
    report = await NodeClassificationBenchmark(num_nodes=16).run()

    metrics = {m.name: m.value for m in report.metrics}
    expected_wins = float(metrics["graph_model_mae"] < metrics["rule_baseline_mae"])
    assert metrics["graph_model_wins"] == expected_wins
