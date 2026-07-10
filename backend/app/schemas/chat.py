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
    # O2 Component 7 (Recommendation Verification): populated from the
    # RecommendationVerification record keyed by this recommendation's id.
    # Defaulted so old callers/tests constructing RecommendationOut without
    # them are unaffected.
    verification_status: str = "unknown"
    verification_reasons: list[str] = []


class ClaimOut(BaseModel):
    """O2 Components 1-3 (Claim Graph / Extraction / Verification): the
    Structured Claim Graph's per-claim record, exposed so the frontend can
    render claims, their verification status, and confidence directly."""

    id: str
    claim_type: str
    text: str
    subject: str
    predicate: str
    object: str = ""
    confidence: float
    propagated_confidence: float | None = None
    source_evidence_ids: list[str] = []
    verification_status: str
    verification_notes: str = ""
    support_count: int = 0
    contradiction_count: int = 0


class SentenceFaithfulnessOut(BaseModel):
    sentence: str
    entailed: bool
    best_claim_id: str | None = None
    score: float


class FaithfulnessOut(BaseModel):
    """O2 Component 8 (Faithfulness Scoring)."""

    overall_faithfulness: float
    sentence_faithfulness: list[SentenceFaithfulnessOut]
    recommendation_faithfulness: float
    graph_faithfulness: float
    claim_coverage: float
    unsupported_claim_count: int
    hallucinated_claim_count: int
    missing_evidence_count: int
    contradiction_rate: float
    summary: str


class SentenceAttributionOut(BaseModel):
    """O2 Component 4 (Evidence Attribution): per-sentence provenance for
    the final answer — which claim, evidence, and Knowledge Graph node
    back each sentence, and the reasoning path connecting them."""

    sentence: str
    supporting_claim_ids: list[str] = []
    supporting_evidence_ids: list[str] = []
    graph_node_ids: list[str] = []
    graph_edge_descriptions: list[str] = []
    confidence: float = 0.0
    reasoning_path: str = ""


class ReflectionReportOut(BaseModel):
    """O3 Component 3 (Self-Reflection Engine): read-only post-answer
    analysis — never influences the answer or confidence themselves."""

    evidence_sufficient: bool
    evidence_sufficiency_note: str
    confidence_justified: bool
    confidence_justification_note: str
    recommendations_verified_ratio: float
    additional_retrieval_suggested: bool
    additional_retrieval_note: str
    summary: str


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
    # O2 additions — all defaulted so pre-O2 clients deserializing this
    # response are unaffected.
    claims: list[ClaimOut] = []
    faithfulness: FaithfulnessOut | None = None
    sentence_attributions: list[SentenceAttributionOut] = []
    # O3 additions — additive/defaulted so pre-O3 clients are unaffected.
    reflection: ReflectionReportOut | None = None
