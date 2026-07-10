"""Counterfactual Simulator (O3.4 Component 5).

`ExecutionTrace` deliberately carries only small per-subsystem summary
dicts, not the original re-runnable domain objects (that would duplicate
the Claim Graph / Evidence Bundle / etc. — exactly the "duplicated
telemetry" O3.3 exists to eliminate). So "replay" here means: apply a
documented, deterministic transform to the relevant subsystem-trace field
(modeling what that field would honestly look like with the subsystem
disabled), then re-run the exact same `ErrorAnalyzer.analyze_trace()`
consumer every real chat turn's failure detection already uses — against
both the original ("baseline") and modified ("simulated") trace — and
compare which failure categories disappear.

Important honesty caveat, documented here rather than hidden: for
toggles that remove a *check* (Verification/NLI/Faithfulness/
Recommendation Verification), "the failure disappeared" means the ALARM
disappeared because the check no longer ran — not that the underlying
data problem was fixed. `CounterfactualResult.note` states this
explicitly for those toggles so a reader doesn't mistake "no longer
detected" for "no longer present."

Never modifies production state: `_apply_toggle` returns a new
`ExecutionTrace` value (frozen dataclass, `dataclasses.replace`); nothing
touches the stored trace or any live subsystem.
"""

import dataclasses
from enum import Enum

from app.error_analysis.analyzer import ErrorAnalyzer
from app.execution.models import ExecutionTrace

_ALARM_ONLY_NOTE = (
    "This toggle removes a verification CHECK, not necessarily the underlying "
    "issue — a disappeared category here means the alarm stopped firing, not "
    "that the original data problem was fixed."
)
_REPLAY_NOTE = "This toggle models the underlying signal the subsystem would have produced if disabled."


class CounterfactualToggle(str, Enum):
    PLANNER = "planner"
    RETRIEVER = "retriever"
    RANKER = "ranker"
    COMPRESSION = "compression"
    VERIFICATION = "verification"
    NLI = "nli"
    RECOMMENDATION_VERIFICATION = "recommendation_verification"
    FAITHFULNESS = "faithfulness"


_ALARM_ONLY_TOGGLES = {
    CounterfactualToggle.VERIFICATION,
    CounterfactualToggle.NLI,
    CounterfactualToggle.FAITHFULNESS,
    CounterfactualToggle.RECOMMENDATION_VERIFICATION,
}

_RANKER_SIMULATED_CONFIDENCE = 0.05


@dataclasses.dataclass(frozen=True)
class CounterfactualResult:
    execution_id: str
    toggle: str
    applicable: bool
    baseline_failure_categories: list[str]
    simulated_failure_categories: list[str]
    disappeared_categories: list[str]
    confidence: float
    expected_improvement: float
    note: str


class CounterfactualSimulator:
    def __init__(self, analyzer: ErrorAnalyzer | None = None):
        self._analyzer = analyzer or ErrorAnalyzer()

    def simulate(self, trace: ExecutionTrace, toggle: CounterfactualToggle) -> CounterfactualResult:
        baseline_categories = sorted({f.category.value for f in self._analyzer.analyze_trace(trace).failures})

        modified_trace = self._apply_toggle(trace, toggle)
        if modified_trace is None:
            return CounterfactualResult(
                execution_id=trace.execution_id, toggle=toggle.value, applicable=False,
                baseline_failure_categories=baseline_categories, simulated_failure_categories=baseline_categories,
                disappeared_categories=[], confidence=0.0, expected_improvement=0.0,
                note=f"Not applicable: this trace has no signal for {toggle.value!r} to toggle.",
            )

        simulated_categories = sorted({f.category.value for f in self._analyzer.analyze_trace(modified_trace).failures})
        disappeared = sorted(set(baseline_categories) - set(simulated_categories))
        expected_improvement = round(len(disappeared) / len(baseline_categories), 4) if baseline_categories else 0.0

        return CounterfactualResult(
            execution_id=trace.execution_id, toggle=toggle.value, applicable=True,
            baseline_failure_categories=baseline_categories, simulated_failure_categories=simulated_categories,
            disappeared_categories=disappeared, confidence=0.6, expected_improvement=expected_improvement,
            note=_ALARM_ONLY_NOTE if toggle in _ALARM_ONLY_TOGGLES else _REPLAY_NOTE,
        )

    @staticmethod
    def _apply_toggle(trace: ExecutionTrace, toggle: CounterfactualToggle) -> ExecutionTrace | None:
        if toggle == CounterfactualToggle.PLANNER:
            if trace.planner_trace is None:
                return None
            node_count = trace.planner_trace.get("node_count", 0)
            return dataclasses.replace(trace, planner_trace={**trace.planner_trace, "failed_node_count": node_count})

        if toggle == CounterfactualToggle.RETRIEVER:
            return dataclasses.replace(trace, retrieval_trace=None)

        if toggle == CounterfactualToggle.RANKER:
            if trace.retrieval_trace is None:
                return None
            return dataclasses.replace(trace, retrieval_trace={**trace.retrieval_trace, "confidence": _RANKER_SIMULATED_CONFIDENCE})

        if toggle == CounterfactualToggle.COMPRESSION:
            # `analyze_trace()` never checks `compression_trace` (Context
            # Compression's failure check — `analyze_compression()` — only
            # runs from the standalone preview endpoint, not the live chat
            # path), so toggling it can never change the simulated outcome.
            return None

        if toggle == CounterfactualToggle.VERIFICATION:
            if trace.verification_trace is None:
                return None
            return dataclasses.replace(trace, verification_trace=None)

        if toggle == CounterfactualToggle.NLI:
            if trace.nli_trace is None:
                return None
            return dataclasses.replace(trace, nli_trace=None)

        if toggle == CounterfactualToggle.RECOMMENDATION_VERIFICATION:
            if trace.recommendation_trace is None:
                return None
            total = trace.recommendation_trace.get("verified_count", 0) + trace.recommendation_trace.get("rejected_count", 0)
            return dataclasses.replace(trace, recommendation_trace={"verified_count": total, "rejected_count": 0})

        if toggle == CounterfactualToggle.FAITHFULNESS:
            if trace.faithfulness_trace is None:
                return None
            return dataclasses.replace(trace, faithfulness_trace=None)

        raise ValueError(f"Unknown counterfactual toggle: {toggle!r}")
