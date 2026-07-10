from dataclasses import dataclass


@dataclass(frozen=True)
class SentenceFaithfulness:
    sentence: str
    entailed: bool
    best_claim_id: str | None
    score: float


@dataclass(frozen=True)
class FaithfulnessReport:
    overall_faithfulness: float
    sentence_faithfulness: list[SentenceFaithfulness]
    recommendation_faithfulness: float
    graph_faithfulness: float
    claim_coverage: float
    unsupported_claim_count: int
    hallucinated_claim_count: int
    missing_evidence_count: int
    contradiction_rate: float
    summary: str = ""
