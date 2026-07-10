from dataclasses import dataclass, field
from enum import Enum


class FailureCategory(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    RANKING_FAILURE = "ranking_failure"
    ENTITY_RESOLUTION_FAILURE = "entity_resolution_failure"
    COMPRESSION_FAILURE = "compression_failure"
    CLAIM_EXTRACTION_FAILURE = "claim_extraction_failure"
    VERIFICATION_FAILURE = "verification_failure"
    RECOMMENDATION_FAILURE = "recommendation_failure"
    NLI_FAILURE = "nli_failure"
    FAITHFULNESS_FAILURE = "faithfulness_failure"
    VISION_FAILURE = "vision_failure"
    OCR_FAILURE = "ocr_failure"
    GRAPH_FAILURE = "graph_failure"
    PLANNER_FAILURE = "planner_failure"
    REASONING_FAILURE = "reasoning_failure"
    PREDICTION_FAILURE = "prediction_failure"
    USER_QUERY_FAILURE = "user_query_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class FailureSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class DetectedFailure:
    category: FailureCategory
    root_cause: str
    affected_components: list[str]
    confidence: float
    suggested_improvement: str
    # O3.2 Component 2 (Failure Taxonomy expansion): severity is derived by
    # the same real signal that triggered the check (see analyzer.py), never
    # guessed. `frequency` defaults to 1 for a freshly-detected, not-yet-
    # persisted failure; `SqlFailureTaxonomyRepository.record()` overwrites
    # it with the real historical count for that category before storing.
    severity: FailureSeverity = FailureSeverity.MEDIUM
    frequency: int = 1


@dataclass(frozen=True)
class ErrorAnalysisReport:
    query: str
    failures: list[DetectedFailure] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return len(self.failures) > 0
