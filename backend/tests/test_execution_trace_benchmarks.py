from app.research.benchmarks.execution_trace_runners import ExecutionTraceBenchmark


async def test_execution_trace_benchmark_reports_real_per_stage_latencies():
    report = await ExecutionTraceBenchmark().run()

    metric_names = {m.name for m in report.metrics}
    assert any(name.startswith("latency_planning_mean_seconds") for name in metric_names)
    assert any(name.startswith("latency_reflection_mean_seconds") for name in metric_names)
    assert "event_volume" in metric_names
    assert "trace_size_bytes" in metric_names


async def test_execution_trace_benchmark_reports_real_instrumentation_overhead():
    report = await ExecutionTraceBenchmark().run()
    metrics_by_name = {m.name: m.value for m in report.metrics}

    assert "instrumentation_overhead_per_call_seconds" in metrics_by_name
    assert "instrumentation_overhead_ratio" in metrics_by_name
    assert metrics_by_name["instrumentation_overhead_ratio"] >= 1.0


async def test_execution_trace_benchmark_reports_real_cache_effectiveness():
    report = await ExecutionTraceBenchmark().run()
    metrics_by_name = {m.name: m.value for m in report.metrics}

    assert metrics_by_name["cache_hit_ratio"] > 0.0
    assert metrics_by_name["cache_size"] == 3.0


async def test_execution_trace_benchmark_event_volume_matches_double_stage_count():
    report = await ExecutionTraceBenchmark().run()
    metrics_by_name = {m.name: m.value for m in report.metrics}

    stage_count = sum(1 for m in report.metrics if m.name.startswith("latency_") and m.name.endswith("_mean_seconds"))
    # Each stage emits one Started + one Finished event.
    assert metrics_by_name["event_volume"] == stage_count * 2


async def test_execution_trace_benchmark_sample_size_matches_run_count():
    report = await ExecutionTraceBenchmark().run()
    assert report.sample_size == 3
    assert report.benchmark_name == "execution_trace_overhead"
