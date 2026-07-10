"""O3.4 (Failure Pattern Mining & Root Cause Intelligence): API response
shapes for FailurePattern, clustering, the Root Cause Graph, counterfactual
simulation, and optimization recommendations."""

from pydantic import BaseModel


class FailurePatternOut(BaseModel):
    pattern_id: str
    category: str
    support: int
    frequency: float
    severity: str
    affected_components: list[str]
    example_execution_ids: list[str]
    confidence: float
    description: str


class FailureAttributionOut(BaseModel):
    pattern_id: str
    primary_cause: str
    secondary_cause: str | None
    contributing_components: list[str]
    evidence_confidence: float
    execution_confidence: float
    severity: str
    historical_frequency: float


class FailureClusterOut(BaseModel):
    cluster_id: str
    strategy_name: str
    key: str
    execution_ids: list[str]
    size: int


class RootCauseNodeOut(BaseModel):
    id: str
    type: str
    label: str
    metadata: dict


class RootCauseEdgeOut(BaseModel):
    source: str
    target: str
    type: str
    confidence: float
    evidence: str


class RootCauseGraphOut(BaseModel):
    nodes: list[RootCauseNodeOut]
    edges: list[RootCauseEdgeOut]


class CounterfactualRequest(BaseModel):
    toggle: str


class CounterfactualResultOut(BaseModel):
    execution_id: str
    toggle: str
    applicable: bool
    baseline_failure_categories: list[str]
    simulated_failure_categories: list[str]
    disappeared_categories: list[str]
    confidence: float
    expected_improvement: float
    note: str


class OptimizationRecommendationOut(BaseModel):
    subsystem: str
    explanation: str
    evidence: list[str]
    supporting_execution_ids: list[str]
    expected_benefit: str
    confidence: float
