from app.research.benchmarks.reasoning_runners import (
    ClaimExtractionBenchmark,
    ClaimVerificationAblationBenchmark,
    ClaimVerificationBenchmark,
    FaithfulnessBenchmark,
    NLIAblationBenchmark,
    NLIBenchmark,
    RecommendationVerificationBenchmark,
)


def _metric(report, name: str) -> float:
    return next(m.value for m in report.metrics if m.name == name)


async def test_claim_extraction_benchmark_recovers_worked_example():
    report = await ClaimExtractionBenchmark().run()
    assert report.benchmark_name == "claim_extraction_quality"
    assert _metric(report, "expected_subject_recall") == 1.0
    assert _metric(report, "no_entity_text_claim_count") == 0.0
    assert _metric(report, "relations_extracted") >= 1.0


async def test_claim_verification_benchmark_reaches_high_accuracy():
    report = await ClaimVerificationBenchmark().run()
    assert report.benchmark_name == "claim_verification_quality"
    assert _metric(report, "accuracy") >= 0.75


async def test_nli_benchmark_reports_overall_and_per_label_accuracy():
    report = await NLIBenchmark().run()
    assert report.benchmark_name == "nli_quality"
    assert _metric(report, "accuracy") >= 0.6
    names = {m.name for m in report.metrics}
    assert "contradiction_accuracy" in names
    assert "entailment_accuracy" in names
    assert "neutral_accuracy" in names


async def test_recommendation_verification_benchmark_reports_precision_recall_f1():
    report = await RecommendationVerificationBenchmark().run()
    assert report.benchmark_name == "recommendation_verification_quality"
    names = {m.name for m in report.metrics}
    assert {"precision", "recall", "f1"}.issubset(names)


async def test_faithfulness_benchmark_ranks_faithful_answer_above_hallucinated():
    report = await FaithfulnessBenchmark().run()
    assert report.benchmark_name == "faithfulness_quality"
    assert _metric(report, "correctly_ranked_faithful_above_hallucinated") == 1.0
    assert _metric(report, "faithful_answer_score") > _metric(report, "hallucinated_answer_score")
    assert _metric(report, "unsupported_claim_count") == 1.0
    assert 0.0 <= _metric(report, "calibration_error_brier") <= 1.0


async def test_claim_verification_ablation_shows_more_unsupported_without_graph_support():
    report = await ClaimVerificationAblationBenchmark().run()
    assert report.benchmark_name == "claim_verification_ablation"
    assert _metric(report, "disabled_unsupported_count") == 0.0
    assert _metric(report, "rule_plus_graph_claims_kept") >= _metric(report, "rule_based_claims_kept")


async def test_nli_ablation_finds_more_contradictions_with_rules_than_disabled():
    report = await NLIAblationBenchmark().run()
    assert report.benchmark_name == "nli_ablation"
    assert _metric(report, "disabled_contradictions_found") == 0.0
    assert _metric(report, "rules_contradictions_found") >= 1.0


async def test_every_reasoning_benchmark_reports_a_real_positive_duration():
    for benchmark_cls in [
        ClaimExtractionBenchmark,
        ClaimVerificationBenchmark,
        NLIBenchmark,
        RecommendationVerificationBenchmark,
        FaithfulnessBenchmark,
        ClaimVerificationAblationBenchmark,
        NLIAblationBenchmark,
    ]:
        report = await benchmark_cls().run()
        assert report.duration_seconds >= 0.0
        assert report.sample_size > 0
        assert report.dataset_name


async def test_claim_extraction_benchmark_dataset_name_and_sample_size():
    report = await ClaimExtractionBenchmark().run()
    assert report.dataset_name == "worked_example_and_negative_case"
    assert report.sample_size == 2


async def test_nli_benchmark_dataset_name_and_sample_size():
    report = await NLIBenchmark().run()
    assert report.dataset_name == "hand_labeled_nli_pairs"
    assert report.sample_size == 9
