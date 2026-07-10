from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReliabilityBucket:
    bucket_range: str
    mean_confidence: float
    accuracy: float | None  # None if the bucket has zero samples
    count: int


@dataclass(frozen=True)
class HistogramBucket:
    bucket_range: str
    count: int


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    reliability_buckets: list[ReliabilityBucket] = field(default_factory=list)
    confidence_histogram: list[HistogramBucket] = field(default_factory=list)
    sample_count: int = 0
