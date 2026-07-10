"""Failure Pattern Mining & Root Cause Intelligence (O3.4).

`FailurePattern` is the only new persisted concept this milestone
introduces — everything else (failures, severities, categories) is read
from `ExecutionTrace` via the existing `ErrorAnalyzer.analyze_trace()`
adapter (O3.3 Component 10), never re-derived independently. No parallel
telemetry.
"""

from dataclasses import dataclass, field

from app.error_analysis.models import FailureSeverity


@dataclass(frozen=True)
class FailurePattern:
    pattern_id: str
    category: str
    support: int
    frequency: float
    severity: FailureSeverity
    affected_components: list[str] = field(default_factory=list)
    example_execution_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""
