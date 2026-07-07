from fastapi import APIRouter, HTTPException, Request

from app.prediction.engine import PredictionEngine
from app.prediction.models import PredictionTaskType
from app.schemas.prediction import PredictionOut, PredictionRequest

router = APIRouter(tags=["predictions"])


@router.post("/predictions", response_model=PredictionOut)
async def create_prediction(payload: PredictionRequest, request: Request) -> PredictionOut:
    try:
        task_type = PredictionTaskType(payload.task_type)
    except ValueError:
        valid = ", ".join(t.value for t in PredictionTaskType)
        raise HTTPException(status_code=422, detail=f"Unknown task_type '{payload.task_type}'. Valid values: {valid}")

    prediction_engine: PredictionEngine = request.app.state.prediction_engine
    prediction = await prediction_engine.predict(payload.equipment_id, task_type)

    return PredictionOut(
        id=prediction.id,
        task_type=prediction.task_type.value,
        equipment_id=prediction.equipment_id,
        value=prediction.value,
        confidence=prediction.confidence,
        explanation=prediction.explanation,
        model_name=prediction.model_name,
    )
