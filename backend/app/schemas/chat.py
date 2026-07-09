from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    page_number: int
    source: str
    entity_ids: list[str]


class EvidenceOut(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page_number: int
    entity_ids: list[str]
    relation_ids: list[str]
    graph_score: float
    vector_score: float
    bm25_score: float
    reranker_score: float
    source: str
    confidence: float
    agent: str
    # O1 Component 5 (Retrieval Explainability): ltr_score mirrors
    # Evidence.ltr_score (see app/retrieval/evidence.py). final_rank/
    # explanation aren't Evidence fields — they depend on the evidence's
    # position within its returned list, so callers compute and pass them
    # in explicitly (see app/retrieval/explain.py).
    ltr_score: float = 0.0
    final_rank: int = 0
    explanation: str = ""


class ContradictionOut(BaseModel):
    conflicting_evidence: list[str]
    confidence: float
    severity: str
    resolution_strategy: str
    recommended_clarification: str
    description: str


class RecommendationOut(BaseModel):
    id: str
    description: str
    priority: str
    severity: str
    estimated_impact: str
    required_personnel: list[str]
    required_equipment: list[str]
    required_documents: list[str]
    estimated_downtime: str
    confidence: float
    justification: str
    supporting_evidence: list[str]


class ChatResponse(BaseModel):
    summary: str
    confidence: float
    citations: list[CitationOut]
    evidence: list[EvidenceOut]
    recommended_actions: list[str]
    reasoning_summary: str
    contradictions: list[ContradictionOut]
    recommendations: list[RecommendationOut]
    latencies: dict[str, float]
