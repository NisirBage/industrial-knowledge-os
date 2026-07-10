from dataclasses import dataclass


@dataclass(frozen=True)
class ReflectionReport:
    evidence_sufficient: bool
    evidence_sufficiency_note: str
    confidence_justified: bool
    confidence_justification_note: str
    recommendations_verified_ratio: float
    additional_retrieval_suggested: bool
    additional_retrieval_note: str
    summary: str
