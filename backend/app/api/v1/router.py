from fastapi import APIRouter

from app.api.v1 import chat, documents, feedback, health, ml, observability, predictions, research, timeline

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
