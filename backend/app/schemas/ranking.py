from pydantic import BaseModel

from app.schemas.ml import FeatureImportanceOut


class RankingTrainRequest(BaseModel):
    sample_size: int = 30
    seed: int = 42
    n_estimators: int = 50


class RankingTrainResponse(BaseModel):
    trained: bool
    algorithm: str
    sample_size: int
    feature_importance: list[FeatureImportanceOut] = []


class RankingStatusOut(BaseModel):
    active_ranker: str
    is_trained: bool
