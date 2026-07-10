from fastapi import APIRouter

from app.api.v1 import (
    chat,
    context_compression,
    documents,
    entity_resolution,
    execution,
    failure_intelligence,
    feedback,
    graph_intelligence,
    health,
    ml,
    observability,
    predictions,
    ranking,
    research,
    timeline,
    vision,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(timeline.router)
api_router.include_router(predictions.router)
api_router.include_router(feedback.router)
api_router.include_router(observability.router)
api_router.include_router(research.router)
api_router.include_router(ml.router)
api_router.include_router(vision.router)
api_router.include_router(graph_intelligence.router)
api_router.include_router(ranking.router)
api_router.include_router(entity_resolution.router)
api_router.include_router(context_compression.router)
api_router.include_router(execution.router)
api_router.include_router(failure_intelligence.router)
