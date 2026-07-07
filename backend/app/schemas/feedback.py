from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    kind: str
    target_id: str
    equipment_id: str | None = None
    reason: str | None = None
    corrected_value: str | None = None


class FeedbackOut(BaseModel):
    id: str
    kind: str
    target_id: str
    equipment_id: str | None
    reason: str | None
    corrected_value: str | None
