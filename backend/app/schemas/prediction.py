from pydantic import BaseModel


class PredictionRequest(BaseModel):
    equipment_id: str
    task_type: str


class PredictionOut(BaseModel):
    id: str
    task_type: str
    equipment_id: str
    value: float | None
    confidence: float
    explanation: str
    model_name: str
