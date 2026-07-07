from fastapi import APIRouter, HTTPException, Request

from app.feedback.engine import FeedbackEngine
from app.feedback.models import FeedbackKind
from app.schemas.feedback import FeedbackOut, FeedbackRequest

router = APIRouter(tags=["feedback"])

_REQUIRES_CORRECTED_VALUE = {
    FeedbackKind.RECOMMENDATION_MODIFIED,
    FeedbackKind.ENTITY_CORRECTION,
    FeedbackKind.RELATION_CORRECTION,
    FeedbackKind.RECOMMENDATION_CORRECTION,
}


@router.post("/feedback", response_model=FeedbackOut)
async def submit_feedback(payload: FeedbackRequest, request: Request) -> FeedbackOut:
    try:
        kind = FeedbackKind(payload.kind)
    except ValueError:
        valid = ", ".join(k.value for k in FeedbackKind)
        raise HTTPException(status_code=422, detail=f"Unknown kind '{payload.kind}'. Valid values: {valid}")

    if kind in _REQUIRES_CORRECTED_VALUE and not payload.corrected_value:
        raise HTTPException(status_code=422, detail=f"'{kind.value}' feedback requires corrected_value")

    engine: FeedbackEngine = request.app.state.feedback_engine

    if kind == FeedbackKind.RECOMMENDATION_ACCEPTED:
        feedback = await engine.accept_recommendation(payload.target_id, payload.equipment_id, payload.reason)
    elif kind == FeedbackKind.RECOMMENDATION_REJECTED:
        feedback = await engine.reject_recommendation(payload.target_id, payload.equipment_id, payload.reason)
    elif kind == FeedbackKind.RECOMMENDATION_MODIFIED:
        feedback = await engine.modify_recommendation(payload.target_id, payload.corrected_value, payload.equipment_id, payload.reason)
    elif kind == FeedbackKind.ENTITY_CORRECTION:
        feedback = await engine.correct_entity(payload.target_id, payload.corrected_value, payload.reason)
    elif kind == FeedbackKind.RELATION_CORRECTION:
        feedback = await engine.correct_relation(payload.target_id, payload.corrected_value, payload.reason)
    else:
        feedback = await engine.correct_recommendation(payload.target_id, payload.corrected_value, payload.reason)

    return FeedbackOut(
        id=feedback.id,
        kind=feedback.kind.value,
        target_id=feedback.target_id,
        equipment_id=feedback.equipment_id,
        reason=feedback.reason,
        corrected_value=feedback.corrected_value,
    )
