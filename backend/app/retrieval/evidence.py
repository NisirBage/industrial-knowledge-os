"""Evidence: the unit every downstream agent (Maintenance, Reasoning, and
future Compliance/Incident/Recommendation agents) consumes. Retrieval used to
hand agents a bag of raw evidence-shaped dicts built straight from search
hits; Evidence makes the score provenance (vector vs BM25 vs graph vs
reranker) and graph linkage (entity_ids/relation_ids) explicit and typed, so
an agent can reason about *why* a piece of evidence was retrieved, not just
its text.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    document_id: str
    page_number: int
    chunk_id: str
    text: str
    entity_ids: list[str] = field(default_factory=list)
    relation_ids: list[str] = field(default_factory=list)
    graph_score: float = 0.0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    reranker_score: float = 0.0
    source: str = "fusion"
    confidence: float = 0.0
    agent: str = ""
    # O1 Component 1: the score actually used to order final results. Equal
    # to reranker_score when no Ranker is injected into HybridRetriever
    # (the default — exact prior behavior), or a Ranker's own output
    # (Reciprocal Rank Fusion / LightGBM LambdaMART) otherwise. Kept
    # separate from `confidence` (always sigmoid(reranker_score)) since a
    # trained ranker's score has no fixed scale to sigmoid meaningfully.
    ltr_score: float = 0.0
