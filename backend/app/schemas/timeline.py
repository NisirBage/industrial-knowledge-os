from pydantic import BaseModel


class TimelineEventOut(BaseModel):
    id: str
    equipment: str
    event_type: str
    description: str
    relation_type: str
    timestamp: str | None


class TimelineOut(BaseModel):
    equipment: str
    view: str
    events: list[TimelineEventOut]
