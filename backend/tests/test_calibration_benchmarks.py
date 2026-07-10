from app.research.benchmarks.calibration_runners import (
    ClaimVerificationCalibrationBenchmark,
    FaithfulnessCalibrationBenchmark,
    RecommendationCalibrationBenchmark,
    RetrievalCalibrationBenchmark,
)


def _metric(report, name: str) -> float:
    return next(m.value for m in report.metrics if m.name == name)


async def test_retrieval_calibration_benchmark_reports_real_metrics():
    report = await RetrievalCalibrationBenchmark().run()
    assert report.benchmark_name == "retrieval_calibration"
    names = {m.name for m in report.metrics}
    assert {"expected_calibration_error", "maximum_calibration_error", "brier_score", "sample_count"}.issubset(names)
    assert _metric(report, "sample_count") > 0


async def test_claim_verification_calibration_benchmark_reports_real_metrics():
    report = await ClaimVerificationCalibrationBenchmark().run()
    assert report.benchmark_name == "claim_verification_calibration"
    assert _metric(report, "sample_count") == 4


async def test_recommendation_calibration_benchmark_reports_real_metrics():
    report = await RecommendationCalibrationBenchmark().run()
    assert report.benchmark_name == "recommendation_calibration"
    assert _metric(report, "sample_count") == 6


async def test_faithfulness_calibration_benchmark_reports_real_metrics():
    report = await FaithfulnessCalibrationBenchmark().run()
    assert report.benchmark_name == "faithfulness_calibration"
    assert _metric(report, "sample_count") > 0
    assert 0.0 <= _metric(report, "brier_score") <= 1.0


async def test_all_calibration_benchmarks_report_bounded_ece_and_mce():
    for benchmark_cls in [
        RetrievalCalibrationBenchmark,
        ClaimVerificationCalibrationBenchmark,
        RecommendationCalibrationBenchmark,
        FaithfulnessCalibrationBenchmark,
    ]:
        report = await benchmark_cls().run()
        assert 0.0 <= _metric(report, "expected_calibration_error") <= 1.0
        assert 0.0 <= _metric(report, "maximum_calibration_error") <= 1.0
        assert _metric(report, "maximum_calibration_error") >= _metric(report, "expected_calibration_error")
