"""FailureEvent (O3.3 Component 5).

A structured failure record that references its originating `ExecutionTrace`
by `execution_id`. This is additive to — not a replacement of — the existing
`DetectedFailure` (`app/error_analysis/models.py`): the spec's "replace
simple failures with structured FailureEvents" is honored by having
`from_detected_failure()` upgrade a `DetectedFailure` into a `FailureEvent`
once an `execution_id` is available, while `ErrorAnalyzer`/`DetectedFailure`
keep their existing signature so no public API breaks.
"""

import uuid
from dataclasses import dataclass, field

from app.error_analysis.models import DetectedFailure, FailureCategory, FailureSeverity


@dataclass(frozen=True)
class FailureEvent:
    id: str
    execution_id: str
    component: str
    failure_type: FailureCategory
    severity: FailureSeverity
    confidence: float
    root_cause: str
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    graph_nodes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def from_detected_failure(
        execution_id: str,
        failure: DetectedFailure,
        *,
        evidence_ids: list[str] | None = None,
        claim_ids: list[str] | None = None,
        graph_nodes: list[str] | None = None,
        metadata: dict | None = None,
    ) -> "FailureEvent":
        return FailureEvent(
            id=str(uuid.uuid4()),
            execution_id=execution_id,
            component=failure.affected_components[0] if failure.affected_components else "unknown",
            failure_type=failure.category,
            severity=failure.severity,
            confidence=failure.confidence,
            root_cause=failure.root_cause,
            evidence_ids=evidence_ids or [],
            claim_ids=claim_ids or [],
            graph_nodes=graph_nodes or [],
            metadata=metadata or {"suggested_improvement": failure.suggested_improvement},
        )
