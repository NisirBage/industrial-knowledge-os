import csv
import io
import json

from app.evaluation.models import NamedMetric
from app.research.benchmarks.models import BenchmarkReport
from app.research.reports.exporter import ReportExporter


def _report() -> BenchmarkReport:
    return BenchmarkReport(
        benchmark_name="entity_extraction",
        metrics=[NamedMetric("precision", 0.9), NamedMetric("recall", 0.8)],
        duration_seconds=1.2345,
        dataset_name="synthetic",
        sample_size=10,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_to_json_round_trips_metrics():
    parsed = json.loads(ReportExporter.to_json(_report()))
    assert parsed["benchmark_name"] == "entity_extraction"
    assert parsed["metrics"][0]["name"] == "precision"
    assert parsed["metrics"][0]["value"] == 0.9


def test_to_csv_has_header_and_rows():
    rows = list(csv.reader(io.StringIO(ReportExporter.to_csv(_report()))))
    assert rows[0] == ["metric", "value"]
    assert rows[1] == ["precision", "0.9"]
    assert rows[2] == ["recall", "0.8"]


def test_to_markdown_includes_table():
    markdown = ReportExporter.to_markdown(_report())
    assert "# Benchmark Report: entity_extraction" in markdown
    assert "| precision | 0.9 |" in markdown
    assert "Sample size: 10" in markdown


def test_to_html_includes_table_rows():
    html = ReportExporter.to_html(_report())
    assert "<h1>Benchmark Report: entity_extraction</h1>" in html
    assert "<td>precision</td><td>0.9</td>" in html


def test_to_json_handles_empty_metrics():
    empty_report = BenchmarkReport("empty", [], 0.0, "none", 0, "2026-01-01T00:00:00+00:00")
    parsed = json.loads(ReportExporter.to_json(empty_report))
    assert parsed["metrics"] == []
