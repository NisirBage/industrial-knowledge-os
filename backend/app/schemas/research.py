from pydantic import BaseModel


class ComponentVersionOut(BaseModel):
    kind: str
    name: str
    version: str
    config: dict[str, str]


class ExperimentOut(BaseModel):
    experiment_id: str
    name: str
    description: str
    configuration: dict[str, str]
    model_versions: list[ComponentVersionOut]
    retriever_versions: list[ComponentVersionOut]
    graph_versions: list[ComponentVersionOut]
    reranker: str
    ontology_version: str
    dataset: str
    timestamp: str
    status: str
    metrics: dict[str, float]
    duration: float | None
    random_seed: int | None
    git_commit: str | None
    error: str | None


class NamedMetricOut(BaseModel):
    name: str
    value: float


class BenchmarkReportOut(BaseModel):
    benchmark_name: str
    metrics: list[NamedMetricOut]
    duration_seconds: float
    dataset_name: str
    sample_size: int
    timestamp: str


class BenchmarkRunRequest(BaseModel):
    sample_size: int = 20
    seed: int = 42


class ABTestRequest(BaseModel):
    sample_size: int = 20
    seed: int = 42


class ABTestResultOut(BaseModel):
    arm_a_name: str
    arm_b_name: str
    metric_name: str
    mean_a: float
    mean_b: float
    mean_delta: float
    t_statistic: float
    p_value: float
    winner: str


class CalibrationBucketOut(BaseModel):
    confidence_bucket: str
    mean_confidence: float
    actual_accuracy: float | None
    sample_count: int


class CalibrationOut(BaseModel):
    buckets: list[CalibrationBucketOut]
