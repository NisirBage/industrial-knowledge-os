"""Automatic Error Analysis (O3 Component 4).

`ErrorAnalyzer.analyze()` takes the same orchestrator-state signals every
chat request already computes (evidence, claim graph, faithfulness report,
recommendation verification, task graph) and checks each against a named,
documented threshold — never a fabricated "the model detected an issue"
claim. A category with no real chat-time signal to check (Compression,
Vision, Graph Intelligence — none of those subsystems are invoked inside
the chat pipeline, the same honest scope note as O2's Claim Verification/
Confidence Propagation gaps) gets its own smaller `analyze_*` method usable
from the endpoint where that subsystem actually runs, rather than a
fabricated always-empty check inside `analyze()`.
"""

from app.claims.models import ClaimGraph
from app.context_compression.models import CompressionStats
from app.error_analysis.models import DetectedFailure, ErrorAnalysisReport, FailureCategory, FailureSeverity
from app.evidence.models import EvidenceBundle
from app.execution.models import ExecutionTrace
from app.faithfulness.models import FaithfulnessReport
from app.planning.models import TaskGraph, TaskStatus
from app.recommendation.verification import VerifiedRecommendationBundle

# Public (not underscore-prefixed): reused by `analyze_trace()` below and by
# `ErrorAnalyzer`'s O3.3 ExecutionTrace-consuming migration adapter, so the
# same named thresholds back both entry points rather than being duplicated.
LOW_EVIDENCE_CONFIDENCE_THRESHOLD = 0.2
HIGH_UNSUPPORTED_CLAIM_RATIO = 0.8
HIGH_CONTRADICTION_RATE = 0.5
LOW_FAITHFULNESS_THRESHOLD = 0.4
AGGRESSIVE_COMPRESSION_RATIO = 0.15


class ErrorAnalyzer:
    def analyze(
        self,
        query: str,
        task_graph: TaskGraph | None,
        evidence_bundle: EvidenceBundle | None,
        claim_graph: ClaimGraph | None,
        faithfulness_report: FaithfulnessReport | None,
        recommendation_verification: VerifiedRecommendationBundle | None,
    ) -> ErrorAnalysisReport:
        failures: list[DetectedFailure] = []

        if planner_failure := self._check_planner(task_graph):
            failures.append(planner_failure)
        if retrieval_failure := self._check_retrieval(evidence_bundle):
            failures.append(retrieval_failure)
        elif ranking_failure := self._check_ranking(evidence_bundle):
            # Only meaningful once we know retrieval itself succeeded —
            # ranking has nothing to rank if there was no evidence at all.
            failures.append(ranking_failure)
        if claim_extraction_failure := self._check_claim_extraction(evidence_bundle, claim_graph):
            failures.append(claim_extraction_failure)
        if verification_failure := self._check_verification(faithfulness_report):
            failures.append(verification_failure)
        if nli_failure := self._check_nli(faithfulness_report):
            failures.append(nli_failure)
        if faithfulness_failure := self._check_faithfulness(faithfulness_report):
            failures.append(faithfulness_failure)
        if recommendation_failure := self._check_recommendation(recommendation_verification):
            failures.append(recommendation_failure)

        return ErrorAnalysisReport(query=query, failures=failures)

    def analyze_compression(self, stats: CompressionStats) -> DetectedFailure | None:
        """Only reachable from the context-compression preview endpoint —
        compression is not invoked inside the live chat pipeline (O1.2
        design), so this cannot be called from `analyze()`."""
        if stats.compression_ratio < AGGRESSIVE_COMPRESSION_RATIO and stats.original_count > 0:
            return DetectedFailure(
                category=FailureCategory.COMPRESSION_FAILURE,
                root_cause=f"Compression ratio {stats.compression_ratio} is unusually aggressive — {stats.original_count} items compressed to {stats.compressed_count}.",
                affected_components=["ContextCompressionEngine"],
                confidence=0.5,
                suggested_improvement="Use a lighter compression level (light/balanced) or verify the near-duplicate threshold isn't over-merging distinct evidence.",
                severity=FailureSeverity.LOW,
            )
        return None

    def analyze_trace(self, trace: ExecutionTrace) -> ErrorAnalysisReport:
        """O3.3 Component 10 migration: the `ExecutionTrace`-consuming entry
        point. Runs the same named-threshold checks as `analyze()`, but
        reads them from the trace's lightweight per-subsystem summary dicts
        (`app/execution/state_adapter.py`) instead of the full
        EvidenceBundle/ClaimGraph/FaithfulnessReport/... instances — proving
        that future subsystems (Pattern Mining, Root Cause Graph,
        Counterfactuals, Optimization Recommendations) can build on
        ExecutionTrace alone. Deeper unification of `analyze()` itself onto
        trace-only inputs is a natural follow-up once every stage populates
        its trace slot; today the two entry points share thresholds but not
        every check's exact object shape."""
        failures: list[DetectedFailure] = []

        planner_trace = trace.planner_trace
        if planner_trace and planner_trace.get("failed_node_count", 0) > 0:
            failed_count = planner_trace["failed_node_count"]
            severity = FailureSeverity.CRITICAL if failed_count == planner_trace.get("node_count") else FailureSeverity.HIGH
            failures.append(DetectedFailure(
                category=FailureCategory.PLANNER_FAILURE,
                root_cause=f"{failed_count} task graph node(s) failed.",
                affected_components=["TaskGraphPlanner", "TaskGraphExecutor"],
                confidence=0.9,
                suggested_improvement="Inspect the failed task node(s) for the underlying retrieval error.",
                severity=severity,
            ))

        retrieval_trace = trace.retrieval_trace
        if retrieval_trace is None or retrieval_trace.get("evidence_count", 0) == 0:
            failures.append(DetectedFailure(
                category=FailureCategory.RETRIEVAL_FAILURE,
                root_cause="No evidence was retrieved for this query.",
                affected_components=["HybridRetriever"],
                confidence=0.9,
                suggested_improvement="Check whether relevant documents have been ingested, or whether the query's entities resolve in the Knowledge Graph.",
                severity=FailureSeverity.CRITICAL,
            ))
        elif retrieval_trace.get("confidence", 1.0) < LOW_EVIDENCE_CONFIDENCE_THRESHOLD:
            failures.append(DetectedFailure(
                category=FailureCategory.RANKING_FAILURE,
                root_cause=f"Evidence was retrieved but fused confidence ({retrieval_trace['confidence']}) is below {LOW_EVIDENCE_CONFIDENCE_THRESHOLD}.",
                affected_components=["EvidenceFusionEngine", "Reranker"],
                confidence=0.6,
                suggested_improvement="Review reranker/ranking model quality for this query type, or check for a sparse/noisy retrieval pool.",
                severity=FailureSeverity.MEDIUM,
            ))

        claim_extraction_trace = trace.claim_extraction_trace
        if (
            retrieval_trace and retrieval_trace.get("evidence_count", 0) > 0
            and (claim_extraction_trace is None or claim_extraction_trace.get("claim_count", 0) == 0)
        ):
            failures.append(DetectedFailure(
                category=FailureCategory.CLAIM_EXTRACTION_FAILURE,
                root_cause="Evidence was retrieved but no claims were extracted from it.",
                affected_components=["ClaimExtractionEngine"],
                confidence=0.5,
                suggested_improvement="Check whether the evidence text contains recognizable entities — regex/GLiNER may have found none.",
                severity=FailureSeverity.MEDIUM,
            ))

        verification_trace = trace.verification_trace
        if (
            verification_trace and verification_trace.get("unsupported_claim_count", 0) > 0
            and verification_trace.get("claim_coverage", 1.0) < (1 - HIGH_UNSUPPORTED_CLAIM_RATIO)
        ):
            coverage = verification_trace["claim_coverage"]
            severity = FailureSeverity.CRITICAL if coverage == 0 else FailureSeverity.HIGH
            failures.append(DetectedFailure(
                category=FailureCategory.VERIFICATION_FAILURE,
                root_cause=f"Only {round(coverage * 100)}% of extracted claims survived verification.",
                affected_components=["ClaimVerificationEngine"],
                confidence=0.6,
                suggested_improvement="Check whether the Knowledge Graph is populated for this query's entities, or whether corroborating evidence is too sparse.",
                severity=severity,
            ))

        nli_trace = trace.nli_trace
        if nli_trace and nli_trace.get("contradiction_rate", 0.0) >= HIGH_CONTRADICTION_RATE:
            rate = nli_trace["contradiction_rate"]
            severity = FailureSeverity.HIGH if rate >= 0.8 else FailureSeverity.MEDIUM
            failures.append(DetectedFailure(
                category=FailureCategory.NLI_FAILURE,
                root_cause=f"{round(rate * 100)}% of claims were flagged as contradicted.",
                affected_components=["ContradictionDetector", "NLIContradictionDetector"],
                confidence=0.5,
                suggested_improvement="Verify whether the underlying evidence is genuinely inconsistent, or whether the contradiction detector is over-triggering on this equipment tag pattern.",
                severity=severity,
            ))

        faithfulness_trace = trace.faithfulness_trace
        if faithfulness_trace and faithfulness_trace.get("overall_faithfulness", 1.0) < LOW_FAITHFULNESS_THRESHOLD:
            score = faithfulness_trace["overall_faithfulness"]
            severity = FailureSeverity.CRITICAL if score < 0.2 else FailureSeverity.HIGH
            failures.append(DetectedFailure(
                category=FailureCategory.FAITHFULNESS_FAILURE,
                root_cause=f"Overall faithfulness ({score}) is below {LOW_FAITHFULNESS_THRESHOLD}.",
                affected_components=["LLMNarrator", "FaithfulnessScorer"],
                confidence=0.7,
                suggested_improvement="Review the Narrator's output against the verified Claim Graph — it may be echoing structural/boilerplate text not grounded in any claim.",
                severity=severity,
            ))

        recommendation_trace = trace.recommendation_trace
        if recommendation_trace:
            total = recommendation_trace.get("verified_count", 0) + recommendation_trace.get("rejected_count", 0)
            if total > 0 and recommendation_trace.get("verified_count", 0) == 0:
                failures.append(DetectedFailure(
                    category=FailureCategory.RECOMMENDATION_FAILURE,
                    root_cause=f"All {total} proposed recommendation(s) were rejected by Recommendation Verification.",
                    affected_components=["RecommendationEngine", "RecommendationVerifier"],
                    confidence=0.6,
                    suggested_improvement="Check whether recommendations cite evidence with real Knowledge Graph support, or whether confidence thresholds are too strict for this query.",
                    severity=FailureSeverity.MEDIUM,
                ))

        return ErrorAnalysisReport(query=trace.query, failures=failures)

    @staticmethod
    def _check_planner(task_graph: TaskGraph | None) -> DetectedFailure | None:
        if task_graph is None:
            return None
        failed_nodes = [n for n in task_graph.nodes if n.status == TaskStatus.FAILED]
        if not failed_nodes:
            return None
        severity = FailureSeverity.CRITICAL if len(failed_nodes) == len(task_graph.nodes) else FailureSeverity.HIGH
        return DetectedFailure(
            category=FailureCategory.PLANNER_FAILURE,
            root_cause=f"{len(failed_nodes)} task graph node(s) failed: {[n.id for n in failed_nodes]}.",
            affected_components=["TaskGraphPlanner", "TaskGraphExecutor"],
            confidence=0.9,
            suggested_improvement="Inspect the failed task node(s) for the underlying retrieval error.",
            severity=severity,
        )

    @staticmethod
    def _check_retrieval(evidence_bundle: EvidenceBundle | None) -> DetectedFailure | None:
        if evidence_bundle is None or evidence_bundle.all_evidence():
            return None
        return DetectedFailure(
            category=FailureCategory.RETRIEVAL_FAILURE,
            root_cause="No evidence was retrieved for this query.",
            affected_components=["HybridRetriever"],
            confidence=0.9,
            suggested_improvement="Check whether relevant documents have been ingested, or whether the query's entities resolve in the Knowledge Graph.",
            severity=FailureSeverity.CRITICAL,
        )

    @staticmethod
    def _check_ranking(evidence_bundle: EvidenceBundle | None) -> DetectedFailure | None:
        if evidence_bundle is None or evidence_bundle.confidence >= LOW_EVIDENCE_CONFIDENCE_THRESHOLD:
            return None
        return DetectedFailure(
            category=FailureCategory.RANKING_FAILURE,
            root_cause=f"Evidence was retrieved but fused confidence ({evidence_bundle.confidence}) is below {LOW_EVIDENCE_CONFIDENCE_THRESHOLD}.",
            affected_components=["EvidenceFusionEngine", "Reranker"],
            confidence=0.6,
            suggested_improvement="Review reranker/ranking model quality for this query type, or check for a sparse/noisy retrieval pool.",
            severity=FailureSeverity.MEDIUM,
        )

    @staticmethod
    def _check_claim_extraction(evidence_bundle: EvidenceBundle | None, claim_graph: ClaimGraph | None) -> DetectedFailure | None:
        if evidence_bundle is None or claim_graph is None:
            return None
        if not evidence_bundle.all_evidence() or claim_graph.claims:
            return None
        return DetectedFailure(
            category=FailureCategory.CLAIM_EXTRACTION_FAILURE,
            root_cause="Evidence was retrieved but no claims were extracted from it.",
            affected_components=["ClaimExtractionEngine"],
            confidence=0.5,
            suggested_improvement="Check whether the evidence text contains recognizable entities — regex/GLiNER may have found none.",
            severity=FailureSeverity.MEDIUM,
        )

    @staticmethod
    def _check_verification(faithfulness_report: FaithfulnessReport | None) -> DetectedFailure | None:
        if faithfulness_report is None or faithfulness_report.unsupported_claim_count == 0:
            return None
        if faithfulness_report.claim_coverage >= (1 - HIGH_UNSUPPORTED_CLAIM_RATIO):
            return None
        severity = FailureSeverity.CRITICAL if faithfulness_report.claim_coverage == 0 else FailureSeverity.HIGH
        return DetectedFailure(
            category=FailureCategory.VERIFICATION_FAILURE,
            root_cause=f"Only {round(faithfulness_report.claim_coverage * 100)}% of extracted claims survived verification.",
            affected_components=["ClaimVerificationEngine"],
            confidence=0.6,
            suggested_improvement="Check whether the Knowledge Graph is populated for this query's entities, or whether corroborating evidence is too sparse.",
            severity=severity,
        )

    @staticmethod
    def _check_nli(faithfulness_report: FaithfulnessReport | None) -> DetectedFailure | None:
        if faithfulness_report is None or faithfulness_report.contradiction_rate < HIGH_CONTRADICTION_RATE:
            return None
        severity = FailureSeverity.HIGH if faithfulness_report.contradiction_rate >= 0.8 else FailureSeverity.MEDIUM
        return DetectedFailure(
            category=FailureCategory.NLI_FAILURE,
            root_cause=f"{round(faithfulness_report.contradiction_rate * 100)}% of claims were flagged as contradicted.",
            affected_components=["ContradictionDetector", "NLIContradictionDetector"],
            confidence=0.5,
            suggested_improvement="Verify whether the underlying evidence is genuinely inconsistent, or whether the contradiction detector is over-triggering on this equipment tag pattern.",
            severity=severity,
        )

    @staticmethod
    def _check_faithfulness(faithfulness_report: FaithfulnessReport | None) -> DetectedFailure | None:
        if faithfulness_report is None or faithfulness_report.overall_faithfulness >= LOW_FAITHFULNESS_THRESHOLD:
            return None
        severity = FailureSeverity.CRITICAL if faithfulness_report.overall_faithfulness < 0.2 else FailureSeverity.HIGH
        return DetectedFailure(
            category=FailureCategory.FAITHFULNESS_FAILURE,
            root_cause=f"Overall faithfulness ({faithfulness_report.overall_faithfulness}) is below {LOW_FAITHFULNESS_THRESHOLD}.",
            affected_components=["LLMNarrator", "FaithfulnessScorer"],
            confidence=0.7,
            suggested_improvement="Review the Narrator's output against the verified Claim Graph — it may be echoing structural/boilerplate text not grounded in any claim.",
            severity=severity,
        )

    @staticmethod
    def _check_recommendation(recommendation_verification: VerifiedRecommendationBundle | None) -> DetectedFailure | None:
        if recommendation_verification is None:
            return None
        total = len(recommendation_verification.verified) + len(recommendation_verification.rejected)
        if total == 0 or recommendation_verification.verified:
            return None
        return DetectedFailure(
            category=FailureCategory.RECOMMENDATION_FAILURE,
            root_cause=f"All {total} proposed recommendation(s) were rejected by Recommendation Verification.",
            affected_components=["RecommendationEngine", "RecommendationVerifier"],
            confidence=0.6,
            suggested_improvement="Check whether recommendations cite evidence with real Knowledge Graph support, or whether confidence thresholds are too strict for this query.",
            severity=FailureSeverity.MEDIUM,
        )
