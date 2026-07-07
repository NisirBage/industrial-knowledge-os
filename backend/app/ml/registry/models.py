from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRecord:
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
