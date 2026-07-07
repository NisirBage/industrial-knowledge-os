from app.evaluation.models import NamedMetric
from app.research.benchmarks.models import BenchmarkReport
from app.research.reports.regression import RegressionDetector


def _report(metrics: list[NamedMetric]) -> BenchmarkReport:
    return BenchmarkReport("entity_extraction", metrics, 1.0, "synthetic", 10, "2026-01-01T00:00:00+00:00")


def test_higher_is_better_metric_drop_flagged_as_regression():
    previous = _report([NamedMetric("f1", 0.9)])
    current = _report([NamedMetric("f1", 0.7)])

    findings = RegressionDetector(threshold=0.05).detect(previous, current)

    assert findings[0].is_regression is True
    assert findings[0].percent_change < 0


def test_higher_is_better_metric_improvement_not_flagged():
    previous = _report([NamedMetric("f1", 0.7)])
    current = _report([NamedMetric("f1", 0.9)])

    findings = RegressionDetector(threshold=0.05).detect(previous, current)

    assert findings[0].is_regression is False


def test_lower_is_better_metric_increase_flagged_as_regression():
    previous = _report([NamedMetric("prediction_latency_seconds", 0.01)])
    current = _report([NamedMetric("prediction_latency_seconds", 0.05)])

    findings = RegressionDetector(threshold=0.05).detect(previous, current)

    assert findings[0].is_regression is True


def test_lower_is_better_metric_decrease_not_flagged():
    previous = _report([NamedMetric("prediction_latency_seconds", 0.05)])
    current = _report([NamedMetric("prediction_latency_seconds", 0.01)])

    findings = RegressionDetector(threshold=0.05).detect(previous, current)

    assert findings[0].is_regression is False


def test_small_change_within_threshold_not_flagged():
    previous = _report([NamedMetric("f1", 0.90)])
    current = _report([NamedMetric("f1", 0.89)])

    findings = RegressionDetector(threshold=0.05).detect(previous, current)

    assert findings[0].is_regression is False


def test_metric_missing_from_previous_report_is_skipped():
    previous = _report([NamedMetric("f1", 0.9)])
    current = _report([NamedMetric("f1", 0.9), NamedMetric("new_metric", 0.5)])

    findings = RegressionDetector().detect(previous, current)

    assert len(findings) == 1
    assert findings[0].metric_name == "f1"


def test_has_regression_true_when_any_finding_is_regression():
    previous = _report([NamedMetric("f1", 0.9)])
    current = _report([NamedMetric("f1", 0.5)])
    findings = RegressionDetector().detect(previous, current)
    assert RegressionDetector.has_regression(findings) is True


def test_has_regression_false_when_no_findings_are_regressions():
    previous = _report([NamedMetric("f1", 0.9)])
    current = _report([NamedMetric("f1", 0.91)])
    findings = RegressionDetector().detect(previous, current)
    assert RegressionDetector.has_regression(findings) is False
