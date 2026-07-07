from fastapi import APIRouter, Request

from app.repositories.learning_repository import LearningRepositoryProtocol
from app.schemas.observability import ObservabilityMetricsOut

router = APIRouter(tags=["observability"])


@router.get("/observability/metrics", response_model=ObservabilityMetricsOut)
async def get_observability_metrics(request: Request) -> ObservabilityMetricsOut:
    learning_repository: LearningRepositoryProtocol = request.app.state.learning_repository

    return ObservabilityMetricsOut(
        feedback_volume=await learning_repository.feedback_volume(),
        average_prediction_confidence=await learning_repository.average_prediction_confidence(),
        average_prediction_latency_seconds=await learning_repository.average_model_accuracy("prediction_latency_seconds"),
        average_feature_generation_latency_seconds=await learning_repository.average_model_accuracy("feature_generation_latency_seconds"),
    )
