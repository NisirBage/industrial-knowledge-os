"""Optimization Recommendation Engine (O3.4 Component 6).

Generates deterministic, human-readable `OptimizationRecommendation`s from
real `FailurePattern`s and `CounterfactualResult`s only — every
recommendation cites the exact pattern/counterfactual that produced it, no
recommendation is emitted without that backing evidence. Advisory only:
nothing here modifies production settings, retrains a model, or auto-applies
any change. Consumers (a human, or a future approval workflow) decide what
to act on.
"""

from dataclasses import dataclass, field

from app.failure_intelligence.counterfactual import CounterfactualResult
from app.failure_intelligence.models import FailurePattern

_PATTERN_TEMPLATES: dict[str, tuple[str, str]] = {
    "retrieval_failure": ("HybridRetriever", "Increase retrieval recall (top_k) or verify ingestion coverage for the affected entities."),
    "ranking_failure": ("Reranker", "Review reranker/ranking thresholds, or retrain the Learning-to-Rank model for this query pattern."),
    "entity_resolution_failure": ("EntityResolver", "Increase the entity merge threshold or review alias coverage for the affected entities."),
    "compression_failure": ("ContextCompressionEngine", "Use a lighter compression level or verify the near-duplicate threshold isn't over-merging distinct evidence."),
    "claim_extraction_failure": ("ClaimExtractionEngine", "Check whether the evidence text for these queries contains recognizable entities."),
    "verification_failure": ("ClaimVerificationEngine", "Expand Knowledge Graph coverage for the affected entities, or lower the verification confidence threshold if coverage is inherently sparse."),
    "recommendation_failure": ("RecommendationVerifier", "Raise the evidence bar recommendations must meet, or review why proposed recommendations lack Knowledge Graph support."),
    "nli_failure": ("ContradictionDetector", "Review whether the NLI/contradiction detector is over-triggering, or whether the evidence is genuinely inconsistent."),
    "faithfulness_failure": ("LLMNarrator", "Review Narrator prompts for boilerplate/unsupported phrasing not grounded in the verified Claim Graph."),
    "planner_failure": ("TaskGraphPlanner", "Inspect why task graph nodes are failing to execute for these queries."),
    "confidence_collapse": ("ConfidencePropagation", "Investigate confidence propagation between retrieval and the final answer — confidence is eroding sharply for these queries."),
    "repeated_failure_sequence": ("Pipeline", "This exact stage sequence recurs across failing executions — investigate the shared upstream conditions."),
}

_TOGGLE_SUBSYSTEM_NAMES: dict[str, str] = {
    "planner": "TaskGraphPlanner",
    "retriever": "HybridRetriever",
    "ranker": "Reranker",
    "compression": "ContextCompressionEngine",
    "verification": "ClaimVerificationEngine",
    "nli": "ContradictionDetector",
    "recommendation_verification": "RecommendationVerifier",
    "faithfulness": "FaithfulnessScorer",
}


@dataclass(frozen=True)
class OptimizationRecommendation:
    subsystem: str
    explanation: str
    evidence: list[str] = field(default_factory=list)
    supporting_execution_ids: list[str] = field(default_factory=list)
    expected_benefit: str = ""
    confidence: float = 0.0


class OptimizationRecommendationEngine:
    def generate(
        self,
        patterns: list[FailurePattern] | None = None,
        counterfactuals: list[CounterfactualResult] | None = None,
    ) -> list[OptimizationRecommendation]:
        recommendations: list[OptimizationRecommendation] = []

        for pattern in patterns or []:
            template = _PATTERN_TEMPLATES.get(pattern.category)
            if template is None:
                continue
            subsystem, advice = template
            recommendations.append(
                OptimizationRecommendation(
                    subsystem=subsystem,
                    explanation=f"{advice} ({pattern.support} execution(s), {round(pattern.frequency * 100)}% of observed traffic).",
                    evidence=[pattern.description],
                    supporting_execution_ids=list(pattern.example_execution_ids),
                    expected_benefit=f"Could address {pattern.support} recurring failure(s) of severity {pattern.severity.value}.",
                    confidence=pattern.confidence,
                )
            )

        for result in counterfactuals or []:
            if not result.applicable or not result.disappeared_categories:
                continue
            subsystem = _TOGGLE_SUBSYSTEM_NAMES.get(result.toggle, result.toggle)
            recommendations.append(
                OptimizationRecommendation(
                    subsystem=subsystem,
                    explanation=(
                        f"Counterfactual analysis on execution {result.execution_id} shows toggling {result.toggle!r} "
                        f"would remove: {', '.join(result.disappeared_categories)}. {result.note}"
                    ),
                    evidence=[f"baseline: {result.baseline_failure_categories}", f"simulated: {result.simulated_failure_categories}"],
                    supporting_execution_ids=[result.execution_id],
                    expected_benefit=f"Estimated {round(result.expected_improvement * 100)}% of this execution's detected failures could be affected.",
                    confidence=result.confidence,
                )
            )

        return recommendations
