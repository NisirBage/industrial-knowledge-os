from dataclasses import dataclass

from app.core.enums import Priority, Severity


@dataclass(frozen=True)
class Recommendation:
    id: str
    description: str
    priority: Priority
    severity: Severity
    estimated_impact: str
    required_personnel: list[str]
    required_equipment: list[str]
    required_documents: list[str]
    estimated_downtime: str
    confidence: float
    justification: str
    supporting_evidence: list[str]  # evidence chunk ids — never empty (see RecommendationEngine)


@dataclass(frozen=True)
class RecommendationBundle:
    recommendations: list[Recommendation]
    overall_confidence: float
