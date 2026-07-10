"""Comprehensive Ablation Framework (O3 Component 1) data model.

Every subsystem the O3 spec names is one `AblationAxis` member. Each axis
has a registered probe (see probes.py) that runs the SAME small synthetic
scenario with the subsystem enabled vs disabled and reports real metrics
for both — never a fabricated "expected" delta.
"""

from dataclasses import dataclass, field
from enum import Enum

from app.evaluation.models import NamedMetric


class AblationAxis(str, Enum):
    PLANNER = "planner"
    PARALLEL_RETRIEVAL = "parallel_retrieval"
    LEARNING_TO_RANK = "learning_to_rank"
    ENTITY_RESOLUTION = "entity_resolution"
    CONTEXT_COMPRESSION = "context_compression"
    CLAIM_EXTRACTION = "claim_extraction"
    CLAIM_VERIFICATION = "claim_verification"
    CONFIDENCE_PROPAGATION = "confidence_propagation"
    NLI = "nli"
    RECOMMENDATION_VERIFICATION = "recommendation_verification"
    FAITHFULNESS_SCORING = "faithfulness_scoring"
    CACHING = "caching"
    EXPLAINABILITY = "explainability"
    VISION = "vision"
    GRAPH_INTELLIGENCE = "graph_intelligence"
    ML_PREDICTION = "ml_prediction"


@dataclass(frozen=True)
class AblationConfiguration:
    """A single point in the ablation space: which axes are enabled.
    Axes not present default to enabled (the production default) — an
    empty set means "everything on," matching today's live pipeline."""

    enabled_axes: frozenset[AblationAxis] = field(default_factory=lambda: frozenset(AblationAxis))

    def is_enabled(self, axis: AblationAxis) -> bool:
        return axis in self.enabled_axes


@dataclass(frozen=True)
class AblationResult:
    axis: AblationAxis
    metrics_enabled: list[NamedMetric]
    metrics_disabled: list[NamedMetric]
    duration_enabled_seconds: float
    duration_disabled_seconds: float
    dataset_name: str
