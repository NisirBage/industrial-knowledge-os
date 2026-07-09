from pydantic import BaseModel

from app.schemas.chat import EvidenceOut


class CompressionRequest(BaseModel):
    query: str
    top_k: int = 20
    level: str = "balanced"


class EvidenceClusterOut(BaseModel):
    cluster_id: str
    cluster_type: str
    representative: EvidenceOut
    member_count: int
    confidence: float


class CompressionStatsOut(BaseModel):
    original_count: int
    compressed_count: int
    exact_duplicate_count: int
    near_duplicate_count: int
    cluster_count: int
    compression_ratio: float
    original_tokens_estimate: int
    compressed_tokens_estimate: int
    token_reduction_ratio: float
    summary: str


class CompressionResponseOut(BaseModel):
    clusters: list[EvidenceClusterOut]
    compressed_evidence: list[EvidenceOut]
    stats: CompressionStatsOut
