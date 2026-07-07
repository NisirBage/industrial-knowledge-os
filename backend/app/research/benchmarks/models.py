import datetime as dt
from dataclasses import dataclass, field
from typing import Protocol

from app.evaluation.models import NamedMetric


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_name: str
    metrics: list[NamedMetric]
    duration_seconds: float
    dataset_name: str
    sample_size: int
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


class BenchmarkRunner(Protocol):
    """All runners are async, even ones with no internal await, so every
    caller (Offline Evaluation Harness, research API routes) can uniformly
    `await runner.run()` without needing to know which runner touches an
    async repository underneath."""

    name: str

    async def run(self) -> BenchmarkReport: ...
