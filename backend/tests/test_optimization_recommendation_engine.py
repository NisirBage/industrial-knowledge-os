from app.error_analysis.models import FailureSeverity
from app.failure_intelligence.counterfactual import CounterfactualResult
from app.failure_intelligence.models import FailurePattern
from app.failure_intelligence.optimization import OptimizationRecommendationEngine


def _pattern(category: str = "retrieval_failure") -> FailurePattern:
    return FailurePattern(
        pattern_id="p1", category=category, support=3, frequency=0.5,
        severity=FailureSeverity.HIGH, affected_components=["HybridRetriever"],
        example_execution_ids=["e1", "e2"], confidence=0.7, description="d",
    )


def _counterfactual(disappeared=None, applicable=True) -> CounterfactualResult:
    return CounterfactualResult(
        execution_id="e1", toggle="verification", applicable=applicable,
        baseline_failure_categories=["verification_failure"], simulated_failure_categories=[],
        disappeared_categories=disappeared if disappeared is not None else ["verification_failure"],
        confidence=0.6, expected_improvement=1.0, note="note",
    )


def test_generate_returns_empty_list_when_nothing_provided():
    assert OptimizationRecommendationEngine().generate() == []


def test_pattern_recommendation_cites_real_supporting_data():
    recommendations = OptimizationRecommendationEngine().generate(patterns=[_pattern()])

    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.subsystem == "HybridRetriever"
    assert rec.supporting_execution_ids == ["e1", "e2"]
    assert rec.evidence == ["d"]
    assert rec.confidence == 0.7


def test_pattern_with_unknown_category_produces_no_recommendation():
    recommendations = OptimizationRecommendationEngine().generate(patterns=[_pattern(category="not_a_known_category")])
    assert recommendations == []


def test_counterfactual_recommendation_cites_real_supporting_data():
    recommendations = OptimizationRecommendationEngine().generate(counterfactuals=[_counterfactual()])

    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.subsystem == "ClaimVerificationEngine"
    assert rec.supporting_execution_ids == ["e1"]
    assert "verification_failure" in rec.explanation


def test_counterfactual_with_no_disappeared_categories_produces_no_recommendation():
    recommendations = OptimizationRecommendationEngine().generate(counterfactuals=[_counterfactual(disappeared=[])])
    assert recommendations == []


def test_non_applicable_counterfactual_produces_no_recommendation():
    recommendations = OptimizationRecommendationEngine().generate(counterfactuals=[_counterfactual(applicable=False)])
    assert recommendations == []


def test_generate_combines_pattern_and_counterfactual_recommendations():
    recommendations = OptimizationRecommendationEngine().generate(patterns=[_pattern()], counterfactuals=[_counterfactual()])
    assert len(recommendations) == 2


def test_every_recommendation_has_nonempty_evidence():
    recommendations = OptimizationRecommendationEngine().generate(patterns=[_pattern()], counterfactuals=[_counterfactual()])
    for rec in recommendations:
        assert rec.evidence
        assert rec.supporting_execution_ids
