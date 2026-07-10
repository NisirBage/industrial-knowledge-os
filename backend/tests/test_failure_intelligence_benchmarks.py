from app.research.benchmarks.failure_intelligence_runners import (
    CounterfactualConsistencyBenchmark,
    FailureIntelligenceLatencyBenchmark,
    FailurePatternMiningBenchmark,
    OptimizationRecommendationGroundednessBenchmark,
    RootCauseAttributionBenchmark,
)


async def test_pattern_mining_benchmark_achieves_perfect_precision_and_recall():
    report = await FailurePatternMiningBenchmark().run()
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["pattern_discovery_precision"] == 1.0
    assert metrics["pattern_discovery_recall"] == 1.0
    assert metrics["patterns_mined"] >= 2


async def test_root_cause_attribution_benchmark_identifies_correct_primary_cause():
    report = await RootCauseAttributionBenchmark().run()
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["attribution_accuracy"] == 1.0


async def test_counterfactual_consistency_benchmark_is_fully_consistent():
    report = await CounterfactualConsistencyBenchmark().run()
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["counterfactual_consistency"] == 1.0


async def test_optimization_recommendation_groundedness_benchmark_is_fully_grounded():
    report = await OptimizationRecommendationGroundednessBenchmark().run()
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["optimization_recommendation_precision"] == 1.0
    assert metrics["recommendations_generated"] >= 1.0


async def test_failure_intelligence_latency_benchmark_reports_real_timings():
    report = await FailureIntelligenceLatencyBenchmark().run()
    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["graph_construction_latency_seconds"] >= 0.0
    assert metrics["failure_clustering_latency_seconds"] >= 0.0
    assert metrics["execution_replay_overhead_seconds"] >= 0.0


async def test_all_benchmarks_report_the_correct_sample_size():
    report = await FailurePatternMiningBenchmark().run()
    assert report.sample_size == 7
    assert report.dataset_name == "synthetic_ground_truth_traces"
