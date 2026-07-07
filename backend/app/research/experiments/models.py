"""Experiment Manager data model (M5 Component 1). Every experiment records
enough to be reproduced later: which versioned components ran, what dataset,
what random seed, and (if available) which git commit — see Reproducibility
in the milestone spec.
"""

from dataclasses import dataclass, field
from enum import Enum

from app.research.model_registry import ComponentVersion


class ExperimentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    name: str
    description: str
    configuration: dict[str, str]
    model_versions: list[ComponentVersion]
    retriever_versions: list[ComponentVersion]
    graph_versions: list[ComponentVersion]
    reranker: str
    ontology_version: str
    dataset: str
    timestamp: str
    status: ExperimentStatus
    metrics: dict[str, float] = field(default_factory=dict)
    duration: float | None = None
    random_seed: int | None = None
    git_commit: str | None = None
    error: str | None = None
