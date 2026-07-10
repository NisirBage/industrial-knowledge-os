"""Failure Pattern Miner (O3.4 Component 1).

Mines `FailurePattern` records from a batch of `ExecutionTrace` objects by
running the exact same `ErrorAnalyzer.analyze_trace()` adapter every
individual chat turn's failure detection already uses (O3.3 Component 10)
against each trace, then aggregating results across the batch. No
independent failure-detection logic is invented here — this module is
purely an aggregation/mining layer over signals `ExecutionTrace` + the
existing Error Analyzer already produce ("consume ExecutionTrace
exclusively, no duplicated telemetry").
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass

from app.error_analysis.analyzer import ErrorAnalyzer
from app.error_analysis.models import DetectedFailure, FailureSeverity
from app.execution.models import ExecutionTrace
from app.execution.timeline import reconstruct_timeline
from app.failure_intelligence.models import FailurePattern

_SEVERITY_RANK = {
    FailureSeverity.LOW: 0,
    FailureSeverity.MEDIUM: 1,
    FailureSeverity.HIGH: 2,
    FailureSeverity.CRITICAL: 3,
}

_MIN_SUPPORT = 2
_CONFIDENCE_COLLAPSE_DROP_THRESHOLD = 0.3
_CONFIDENCE_COLLAPSE_CATEGORY = "confidence_collapse"
_REPEATED_SEQUENCE_CATEGORY = "repeated_failure_sequence"


@dataclass(frozen=True)
class _ConfidenceCollapseSignal:
    """Not a `DetectedFailure`/`FailureCategory` — confidence collapse is a
    cross-stage trend, not one of the Failure Taxonomy's per-subsystem
    categories, so it's tracked as its own lightweight signal (same shape
    `_build_pattern` needs: severity/affected_components/confidence)
    rather than misusing an unrelated existing category."""

    severity: FailureSeverity
    affected_components: list[str]
    confidence: float


class FailurePatternMiner:
    def __init__(self, analyzer: ErrorAnalyzer | None = None, min_support: int = _MIN_SUPPORT):
        self._analyzer = analyzer or ErrorAnalyzer()
        self._min_support = min_support

    def mine(self, traces: list[ExecutionTrace]) -> list[FailurePattern]:
        if not traces:
            return []

        entries_by_category: dict[str, list[tuple[str, DetectedFailure | _ConfidenceCollapseSignal]]] = defaultdict(list)
        traces_with_failures: list[ExecutionTrace] = []

        for trace in traces:
            report = self._analyzer.analyze_trace(trace)
            if report.has_failures:
                traces_with_failures.append(trace)
            for failure in report.failures:
                entries_by_category[failure.category.value].append((trace.execution_id, failure))

            collapse = self._detect_confidence_collapse(trace)
            if collapse is not None:
                entries_by_category[_CONFIDENCE_COLLAPSE_CATEGORY].append((trace.execution_id, collapse))

        patterns = [
            self._build_pattern(category, entries, len(traces))
            for category, entries in entries_by_category.items()
            if len(entries) >= self._min_support
        ]

        sequence_pattern = self._mine_repeated_sequence(traces_with_failures, len(traces))
        if sequence_pattern is not None:
            patterns.append(sequence_pattern)

        return patterns

    @staticmethod
    def _detect_confidence_collapse(trace: ExecutionTrace) -> _ConfidenceCollapseSignal | None:
        """A real, computed cross-check: if retrieval surfaced reasonable
        confidence but the FINAL answer confidence collapsed well below it,
        something downstream eroded confidence sharply — the same kind of
        cross-stage confidence check `SelfReflectionEngine` already
        performs (confidence vs. faithfulness), applied across the whole
        trace's confidence trajectory instead."""
        retrieval_trace = trace.retrieval_trace
        if retrieval_trace is None:
            return None
        retrieval_confidence = retrieval_trace.get("confidence")
        if retrieval_confidence is None:
            return None
        drop = round(retrieval_confidence - trace.confidence, 4)
        if drop < _CONFIDENCE_COLLAPSE_DROP_THRESHOLD:
            return None
        return _ConfidenceCollapseSignal(
            severity=FailureSeverity.HIGH if drop >= 0.5 else FailureSeverity.MEDIUM,
            affected_components=["ConfidencePropagation", "Narrator"],
            confidence=drop,
        )

    def _mine_repeated_sequence(self, traces_with_failures: list[ExecutionTrace], total_traces: int) -> FailurePattern | None:
        sequence_execution_ids: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for trace in traces_with_failures:
            sequence = tuple(reconstruct_timeline(trace).component_order)
            if sequence:
                sequence_execution_ids[sequence].append(trace.execution_id)

        if not sequence_execution_ids:
            return None

        most_common_sequence, execution_ids = max(sequence_execution_ids.items(), key=lambda item: len(item[1]))
        if len(execution_ids) < self._min_support:
            return None

        return FailurePattern(
            pattern_id=str(uuid.uuid4()),
            category=_REPEATED_SEQUENCE_CATEGORY,
            support=len(execution_ids),
            frequency=round(len(execution_ids) / total_traces, 4),
            severity=FailureSeverity.MEDIUM,
            affected_components=list(most_common_sequence),
            example_execution_ids=execution_ids[:5],
            confidence=round(len(execution_ids) / len(traces_with_failures), 4),
            description=f"{len(execution_ids)} failing execution(s) shared the exact same component sequence: {' -> '.join(most_common_sequence)}.",
        )

    @staticmethod
    def _build_pattern(category: str, entries: list[tuple[str, "DetectedFailure | _ConfidenceCollapseSignal"]], total_traces: int) -> FailurePattern:
        execution_ids = [execution_id for execution_id, _ in entries]
        signals = [signal for _, signal in entries]
        max_severity = max((s.severity for s in signals), key=lambda s: _SEVERITY_RANK[s])
        affected_components = sorted({component for s in signals for component in s.affected_components})
        mean_confidence = round(sum(s.confidence for s in signals) / len(signals), 4)

        return FailurePattern(
            pattern_id=str(uuid.uuid4()),
            category=category,
            support=len(entries),
            frequency=round(len(entries) / total_traces, 4),
            severity=max_severity,
            affected_components=affected_components,
            example_execution_ids=execution_ids[:5],
            confidence=mean_confidence,
            description=f"{category} occurred in {len(entries)} of {total_traces} execution(s) ({round(len(entries) / total_traces * 100)}%).",
        )
