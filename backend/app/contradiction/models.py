"""Contradiction data model. Deliberately dependency-free (no import of
EvidenceBundle or anything else) so app.evidence.models can reference
Contradiction without creating a circular import — only app.contradiction.detector
(the detection logic) depends on the evidence models, one direction only.
"""

from dataclasses import dataclass, field

from app.core.enums import Severity


@dataclass(frozen=True)
class Contradiction:
    conflicting_evidence: list[str]  # chunk ids
    confidence: float
    severity: Severity
    resolution_strategy: str
    recommended_clarification: str
    description: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
