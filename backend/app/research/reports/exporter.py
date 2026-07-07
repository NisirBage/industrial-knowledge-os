"""Report Generation (M5): export a BenchmarkReport as JSON, CSV, Markdown,
or HTML. PDF export is a documented future extension (per the M5 spec) —
not implemented here since it needs a rendering dependency this milestone
doesn't otherwise require.
"""

import csv
import dataclasses
import io
import json

from app.research.benchmarks.models import BenchmarkReport


class ReportExporter:
    @staticmethod
    def to_json(report: BenchmarkReport) -> str:
        return json.dumps(dataclasses.asdict(report), indent=2)

    @staticmethod
    def to_csv(report: BenchmarkReport) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["metric", "value"])
        for metric in report.metrics:
            writer.writerow([metric.name, metric.value])
        return buffer.getvalue()

    @staticmethod
    def to_markdown(report: BenchmarkReport) -> str:
        lines = [
            f"# Benchmark Report: {report.benchmark_name}",
            "",
            f"- Dataset: {report.dataset_name}",
            f"- Sample size: {report.sample_size}",
            f"- Duration: {report.duration_seconds:.4f}s",
            f"- Timestamp: {report.timestamp}",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        lines += [f"| {metric.name} | {metric.value} |" for metric in report.metrics]
        return "\n".join(lines)

    @staticmethod
    def to_html(report: BenchmarkReport) -> str:
        rows = "".join(f"<tr><td>{metric.name}</td><td>{metric.value}</td></tr>" for metric in report.metrics)
        return (
            "<html><body>"
            f"<h1>Benchmark Report: {report.benchmark_name}</h1>"
            f"<p>Dataset: {report.dataset_name} | Sample size: {report.sample_size} | "
            f"Duration: {report.duration_seconds:.4f}s | Timestamp: {report.timestamp}</p>"
            f"<table border=\"1\"><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
            "</body></html>"
        )
