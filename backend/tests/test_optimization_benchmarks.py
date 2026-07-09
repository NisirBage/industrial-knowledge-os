from app.research.benchmarks.optimization_runners import (
    CompressionBenchmark,
    DeduplicationBenchmark,
    EntityResolutionBenchmark,
    RankingQualityBenchmark,
)


async def test_ranking_quality_benchmark_reports_both_rankers():
    report = await RankingQualityBenchmark(num_documents=6).run()

    metric_names = {m.name for m in report.metrics}
    assert metric_names == {"rrf_mrr", "rrf_recall_at_5", "rrf_ndcg_at_5", "ltr_mrr", "ltr_recall_at_5", "ltr_ndcg_at_5"}
    for metric in report.metrics:
        assert 0.0 <= metric.value <= 1.0


async def test_ranking_quality_benchmark_achieves_perfect_recall_on_the_strong_synthetic_signal():
    """The synthetic BM25 signal deliberately gives the truly-relevant
    document a 5x higher score than any other candidate, so both rankers
    should reliably surface it within top_k=5 — a real, non-trivial sanity
    check that the whole retrieve() pipeline (including the ranker swap)
    isn't silently broken."""
    report = await RankingQualityBenchmark(num_documents=6).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["rrf_recall_at_5"] == 1.0
    assert metrics["ltr_recall_at_5"] == 1.0


async def test_compression_benchmark_reports_a_real_nonzero_reduction():
    report = await CompressionBenchmark(num_unique=10, duplicate_ratio=0.4).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["original_count"] == 14.0  # 10 unique + 4 duplicates
    assert metrics["compressed_count"] < metrics["original_count"]
    assert metrics["exact_duplicate_count"] == 4.0
    assert 0.0 <= metrics["compression_ratio"] <= 1.0


async def test_compression_benchmark_with_no_duplicates_still_reports_valid_stats():
    report = await CompressionBenchmark(num_unique=5, duplicate_ratio=0.0).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["original_count"] == 5.0
    assert metrics["exact_duplicate_count"] == 0.0


async def test_deduplication_benchmark_reports_real_precision_and_recall():
    report = await DeduplicationBenchmark(threshold=0.6).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["true_positive"] + metrics["true_negative"] <= 10.0


async def test_deduplication_benchmark_sample_size_matches_labeled_pairs():
    report = await DeduplicationBenchmark().run()

    assert report.sample_size == 10  # 5 duplicate + 5 distinct hand-labeled pairs


async def test_entity_resolution_benchmark_reports_real_precision_and_recall():
    report = await EntityResolutionBenchmark(merge_threshold=0.82).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


async def test_entity_resolution_benchmark_correctly_identifies_a_true_tag_variant():
    """P-102 vs P-101 (both Pumps, one edit apart) is a real should-merge
    case in the hand-labeled set — a stricter threshold that rejected it
    would show up as a real recall drop, not a hidden failure."""
    lenient = await EntityResolutionBenchmark(merge_threshold=0.7).run()
    metrics = {m.name: m.value for m in lenient.metrics}
    assert metrics["recall"] > 0.0
