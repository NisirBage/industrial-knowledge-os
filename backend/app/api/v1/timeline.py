from fastapi import APIRouter, Query, Request

from app.schemas.timeline import TimelineEventOut, TimelineOut
from app.timeline.engine import TimelineEngine

router = APIRouter(tags=["timeline"])

_VIEWS = {
    "full": lambda engine, name: engine.full_history(name),
    "equipment": lambda engine, name: engine.equipment_timeline(name),
    "incident": lambda engine, name: engine.incident_timeline(name),
    "maintenance": lambda engine, name: engine.maintenance_timeline(name),
}


@router.get("/equipment/{canonical_name}/timeline", response_model=TimelineOut)
def get_equipment_timeline(canonical_name: str, request: Request, view: str = Query("full", pattern="^(full|equipment|incident|maintenance)$")) -> TimelineOut:
    timeline_engine: TimelineEngine = request.app.state.timeline_engine
    events = _VIEWS[view](timeline_engine, canonical_name)
    return TimelineOut(
        equipment=canonical_name,
        view=view,
        events=[
            TimelineEventOut(
                id=e.id,
                equipment=e.equipment,
                event_type=e.event_type.value,
                description=e.description,
                relation_type=e.relation_type,
                timestamp=e.timestamp,
            )
            for e in events
        ],
    )
