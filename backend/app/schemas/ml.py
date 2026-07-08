from pydantic import BaseModel


class MLTrainRequest(BaseModel):
    sample_size: int = 300
    seed: int = 42
    # v1.0.1: off by default (extra training cost, k additional model fits) —
    # opt in when you want a variance estimate on top of the held-out split.
    include_cross_validation: bool = False
    cv_folds: int = 5


class FeatureImportanceOut(BaseModel):
    feature_name: str
    importance: float


class CalibrationPointOut(BaseModel):
    predicted_probability: float
    observed_frequency: float


class RocPointOut(BaseModel):
    fpr: float
    tpr: float


class ModelRecordOut(BaseModel):
    model_id: str
    task_type: str
    algorithm: str
    training_data_hash: str
    hyperparameters: dict[str, str]
    metrics: dict[str, float]
    feature_schema: list[str]
    timestamp: str
    author: str
    experiment_id: str | None
    serialization_path: str
    serialization_format: str


class MLTrainingResultOut(BaseModel):
    model: ModelRecordOut
    feature_importance: list[FeatureImportanceOut]
    calibration_curve: list[CalibrationPointOut]
    confusion_matrix: list[list[int]] | None
    roc_curve: list[RocPointOut]
    sample_size: int
    duration_seconds: float
    # Distinct from `model.metrics` (a single held-out train/test split):
    # the mean of the same metrics across `cv_folds` independent folds, a
    # variance-aware complement to the point estimate. `None` unless the
    # caller opted in via `include_cross_validation`.
    cross_validation_metrics: dict[str, float] | None = None
    cross_validation_fold_count: int | None = None


class DriftRequest(BaseModel):
    baseline: list[float]
    recent: list[float]


class DriftResultOut(BaseModel):
    psi: float
    severity: str


class FeatureDefinitionOut(BaseModel):
    name: str
    description: str
    source: str
    owner: str
    datatype: str
    version: str
    generation_time: str
