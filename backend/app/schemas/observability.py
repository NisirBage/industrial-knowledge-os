from pydantic import BaseModel


class ObservabilityMetricsOut(BaseModel):
    feedback_volume: int
    average_prediction_confidence: float | None
    average_prediction_latency_seconds: float | None
    average_feature_generation_latency_seconds: float | None


class CacheStatsOut(BaseModel):
    name: str
    hits: int
    misses: int
    evictions: int
    size: int
    max_size: int
    hit_ratio: float
