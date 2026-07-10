"""Failure Attribution Engine (O3.4 Component 4).

For every `FailurePattern`, computes a `FailureAttribution` using only the
pattern's own real, already-computed fields (`affected_components`,
`confidence`, `severity`, `frequency`) plus the real `confidence` of the
`ExecutionTrace`s it names in `example_execution_ids` — never a fabricated
cause. When a pattern has no `affected_components` (possible for the
cross-stage "confidence_collapse" pattern), the primary cause is reported
as "unattributed" with zero evidence confidence rather than guessing.
"""

from dataclasses import dataclass, field

from app.error_analysis.models import FailureSeverity
from app.execution.models import ExecutionTrace
from app.failure_intelligence.models import FailurePattern

_UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class FailureAttribution:
    pattern_id: str
    primary_cause: str
    secondary_cause: str | None
    contributing_components: list[str] = field(default_factory=list)
    evidence_confidence: float = 0.0
    execution_confidence: float = 0.0
    severity: FailureSeverity = FailureSeverity.MEDIUM
    historical_frequency: float = 0.0


class FailureAttributionEngine:
    def attribute(self, pattern: FailurePattern, traces: list[ExecutionTrace] | None = None) -> FailureAttribution:
        components = pattern.affected_components
        primary_cause = components[0] if components else _UNATTRIBUTED
        secondary_cause = components[1] if len(components) > 1 else None
        contributing_components = components[2:] if len(components) > 2 else []
        evidence_confidence = pattern.confidence if components else 0.0

        execution_confidence = self._mean_execution_confidence(pattern, traces or [])

        return FailureAttribution(
            pattern_id=pattern.pattern_id,
            primary_cause=primary_cause,
            secondary_cause=secondary_cause,
            contributing_components=contributing_components,
            evidence_confidence=evidence_confidence,
            execution_confidence=execution_confidence,
            severity=pattern.severity,
            historical_frequency=pattern.frequency,
        )

    @staticmethod
    def _mean_execution_confidence(pattern: FailurePattern, traces: list[ExecutionTrace]) -> float:
        matching = [t.confidence for t in traces if t.execution_id in pattern.example_execution_ids]
        if not matching:
            return 0.0
        return round(sum(matching) / len(matching), 4)
