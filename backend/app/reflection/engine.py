"""Self-Reflection Engine (O3 Component 3).

Runs last in the pipeline (after Faithfulness Scoring) and never changes
the answer — it only evaluates the same real signals every other O2/O3
stage already computed (evidence bundle, final confidence, faithfulness
report, recommendation verification) and writes a `ReflectionReport` purely
for analysis. Nothing downstream reads this report to alter behavior; it
exists for the Research Dashboard and Failure Taxonomy, not the Narrator.
"""

from app.evidence.models import EvidenceBundle
from app.execution.models import ExecutionTrace
from app.faithfulness.models import FaithfulnessReport
from app.recommendation.verification import VerifiedRecommendationBundle
from app.reflection.models import ReflectionReport

# Public (not underscore-prefixed): reused by `reflect_trace()` below, the
# O3.3 ExecutionTrace-consuming migration adapter, so both entry points
# share the same named thresholds rather than duplicating them.
MIN_SUFFICIENT_EVIDENCE_COUNT = 2
HIGH_CONFIDENCE_THRESHOLD = 0.7
LOW_FAITHFULNESS_THRESHOLD = 0.4
LOW_EVIDENCE_COUNT_FOR_MORE_RETRIEVAL = 3


class SelfReflectionEngine:
    def reflect(
        self,
        query: str,
        evidence_bundle: EvidenceBundle,
        confidence: float,
        faithfulness_report: FaithfulnessReport | None,
        recommendation_verification: VerifiedRecommendationBundle | None,
    ) -> ReflectionReport:
        evidence_count = len(evidence_bundle.all_evidence())

        evidence_sufficient = evidence_count >= MIN_SUFFICIENT_EVIDENCE_COUNT and not evidence_bundle.missing_information
        evidence_note = (
            f"{evidence_count} evidence item(s), {len(evidence_bundle.missing_information)} known gap(s)."
            if not evidence_sufficient
            else f"{evidence_count} evidence item(s), no known gaps."
        )

        confidence_justified, confidence_note = self._check_confidence_justified(confidence, faithfulness_report)

        verified_ratio = self._recommendations_verified_ratio(recommendation_verification)

        additional_retrieval_suggested = evidence_count < LOW_EVIDENCE_COUNT_FOR_MORE_RETRIEVAL or bool(evidence_bundle.missing_information)
        retrieval_note = (
            f"Only {evidence_count} evidence item(s) retrieved — additional retrieval may surface more support."
            if additional_retrieval_suggested
            else "Evidence volume appears adequate."
        )

        summary = self._summary(evidence_sufficient, confidence_justified, verified_ratio, additional_retrieval_suggested)

        return ReflectionReport(
            evidence_sufficient=evidence_sufficient,
            evidence_sufficiency_note=evidence_note,
            confidence_justified=confidence_justified,
            confidence_justification_note=confidence_note,
            recommendations_verified_ratio=verified_ratio,
            additional_retrieval_suggested=additional_retrieval_suggested,
            additional_retrieval_note=retrieval_note,
            summary=summary,
        )

    def reflect_trace(self, trace: ExecutionTrace) -> ReflectionReport:
        """O3.3 Component 10 migration: the `ExecutionTrace`-consuming entry
        point. Same checks as `reflect()`, read from the trace's lightweight
        retrieval/faithfulness/recommendation summary dicts
        (`app/execution/state_adapter.py`) instead of the full
        EvidenceBundle/FaithfulnessReport/VerifiedRecommendationBundle
        instances."""
        retrieval_trace = trace.retrieval_trace or {}
        evidence_count = retrieval_trace.get("evidence_count", 0)
        missing_information_count = retrieval_trace.get("missing_information_count", 0)

        evidence_sufficient = evidence_count >= MIN_SUFFICIENT_EVIDENCE_COUNT and missing_information_count == 0
        evidence_note = (
            f"{evidence_count} evidence item(s), {missing_information_count} known gap(s)."
            if not evidence_sufficient
            else f"{evidence_count} evidence item(s), no known gaps."
        )

        faithfulness_trace = trace.faithfulness_trace
        overall_faithfulness = faithfulness_trace.get("overall_faithfulness") if faithfulness_trace else None
        confidence_justified, confidence_note = self._check_confidence_justified_from_trace(trace.confidence, overall_faithfulness)

        verified_ratio = self._recommendations_verified_ratio_from_trace(trace.recommendation_trace)

        additional_retrieval_suggested = evidence_count < LOW_EVIDENCE_COUNT_FOR_MORE_RETRIEVAL or missing_information_count > 0
        retrieval_note = (
            f"Only {evidence_count} evidence item(s) retrieved — additional retrieval may surface more support."
            if additional_retrieval_suggested
            else "Evidence volume appears adequate."
        )

        summary = self._summary(evidence_sufficient, confidence_justified, verified_ratio, additional_retrieval_suggested)

        return ReflectionReport(
            evidence_sufficient=evidence_sufficient,
            evidence_sufficiency_note=evidence_note,
            confidence_justified=confidence_justified,
            confidence_justification_note=confidence_note,
            recommendations_verified_ratio=verified_ratio,
            additional_retrieval_suggested=additional_retrieval_suggested,
            additional_retrieval_note=retrieval_note,
            summary=summary,
        )

    @staticmethod
    def _check_confidence_justified_from_trace(confidence: float, overall_faithfulness: float | None) -> tuple[bool, str]:
        if overall_faithfulness is None:
            return True, "No faithfulness report available to cross-check confidence against."
        if confidence >= HIGH_CONFIDENCE_THRESHOLD and overall_faithfulness < LOW_FAITHFULNESS_THRESHOLD:
            return False, (
                f"Reported confidence ({confidence}) is high but overall faithfulness "
                f"({overall_faithfulness}) is low — the answer may be more confident than its grounding supports."
            )
        return True, f"Confidence ({confidence}) is consistent with overall faithfulness ({overall_faithfulness})."

    @staticmethod
    def _recommendations_verified_ratio_from_trace(recommendation_trace: dict | None) -> float:
        if not recommendation_trace:
            return 1.0
        total = recommendation_trace.get("verified_count", 0) + recommendation_trace.get("rejected_count", 0)
        if total == 0:
            return 1.0
        return round(recommendation_trace.get("verified_count", 0) / total, 4)

    @staticmethod
    def _check_confidence_justified(confidence: float, faithfulness_report: FaithfulnessReport | None) -> tuple[bool, str]:
        if faithfulness_report is None:
            return True, "No faithfulness report available to cross-check confidence against."
        if confidence >= HIGH_CONFIDENCE_THRESHOLD and faithfulness_report.overall_faithfulness < LOW_FAITHFULNESS_THRESHOLD:
            return False, (
                f"Reported confidence ({confidence}) is high but overall faithfulness "
                f"({faithfulness_report.overall_faithfulness}) is low — the answer may be more confident than its grounding supports."
            )
        return True, f"Confidence ({confidence}) is consistent with overall faithfulness ({faithfulness_report.overall_faithfulness})."

    @staticmethod
    def _recommendations_verified_ratio(recommendation_verification: VerifiedRecommendationBundle | None) -> float:
        if recommendation_verification is None:
            return 1.0
        total = len(recommendation_verification.verified) + len(recommendation_verification.rejected)
        if total == 0:
            return 1.0
        return round(len(recommendation_verification.verified) / total, 4)

    @staticmethod
    def _summary(evidence_sufficient: bool, confidence_justified: bool, verified_ratio: float, additional_retrieval_suggested: bool) -> str:
        parts = []
        parts.append("Evidence sufficient." if evidence_sufficient else "Evidence may be insufficient.")
        parts.append("Confidence appears justified." if confidence_justified else "Confidence may be overstated relative to faithfulness.")
        parts.append(f"{round(verified_ratio * 100)}% of recommendations passed verification.")
        if additional_retrieval_suggested:
            parts.append("Additional retrieval could help.")
        return " ".join(parts)
