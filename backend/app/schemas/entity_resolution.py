from pydantic import BaseModel


class DuplicateCandidateOut(BaseModel):
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    confidence: float
    merge_reason: str


class MergeRequest(BaseModel):
    source_id: str
    target_id: str
    reason: str = "manual"


class MergeResponseOut(BaseModel):
    merge_id: str
    source_id: str
    target_id: str
    reason: str
    timestamp: str


class RollbackRequest(BaseModel):
    merge_id: str


class RollbackResponseOut(BaseModel):
    restored_node_id: str
