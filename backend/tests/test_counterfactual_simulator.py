from app.execution.models import ExecutionTrace
from app.failure_intelligence.counterfactual import CounterfactualSimulator, CounterfactualToggle


def _healthy_retrieval_and_claims() -> dict:
    return {
        "retrieval_trace": {"evidence_count": 2, "confidence": 0.8, "missing_information_count": 0},
        "claim_extraction_trace": {"claim_count": 1},
    }


def test_verification_toggle_removes_the_verification_failure_alarm():
    trace = ExecutionTrace(
        execution_id="e1", query="q", confidence=0.5,
        verification_trace={"claim_coverage": 0.0, "unsupported_claim_count": 9},
        **_healthy_retrieval_and_claims(),
    )
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.VERIFICATION)

    assert result.applicable is True
    assert "verification_failure" in result.baseline_failure_categories
    assert "verification_failure" not in result.simulated_failure_categories
    assert "verification_failure" in result.disappeared_categories
    assert result.expected_improvement == 1.0
    assert "alarm" in result.note.lower()


def test_nli_toggle_removes_the_nli_failure_alarm():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.5, nli_trace={"contradiction_rate": 0.9})
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.NLI)

    assert "nli_failure" in result.disappeared_categories


def test_faithfulness_toggle_removes_the_faithfulness_failure_alarm():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.5, faithfulness_trace={"overall_faithfulness": 0.1})
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.FAITHFULNESS)

    assert "faithfulness_failure" in result.disappeared_categories


def test_recommendation_verification_toggle_removes_the_recommendation_failure():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.5, recommendation_trace={"verified_count": 0, "rejected_count": 2})
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.RECOMMENDATION_VERIFICATION)

    assert "recommendation_failure" in result.disappeared_categories


def test_planner_toggle_introduces_planner_failure_when_baseline_healthy():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8, planner_trace={"node_count": 2, "failed_node_count": 0, "task_type": "general_retrieval"})
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.PLANNER)

    assert "planner_failure" not in result.baseline_failure_categories
    assert "planner_failure" in result.simulated_failure_categories


def test_ranker_toggle_introduces_ranking_failure_when_baseline_healthy():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8, retrieval_trace={"evidence_count": 3, "confidence": 0.9, "missing_information_count": 0}, claim_extraction_trace={"claim_count": 1})
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.RANKER)

    assert "ranking_failure" not in result.baseline_failure_categories
    assert "ranking_failure" in result.simulated_failure_categories


def test_compression_toggle_is_never_applicable():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8)
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.COMPRESSION)

    assert result.applicable is False
    assert result.confidence == 0.0
    assert result.expected_improvement == 0.0


def test_planner_toggle_not_applicable_when_no_planner_trace():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8)
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.PLANNER)
    assert result.applicable is False


def test_ranker_toggle_not_applicable_when_no_retrieval_trace():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8)
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.RANKER)
    assert result.applicable is False


def test_retriever_toggle_is_always_applicable():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.8)
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.RETRIEVER)
    assert result.applicable is True


def test_expected_improvement_is_zero_when_baseline_has_no_failures():
    trace = ExecutionTrace(
        execution_id="e1", query="q", confidence=0.8,
        verification_trace={"claim_coverage": 1.0, "unsupported_claim_count": 0},
        **_healthy_retrieval_and_claims(),
    )
    result = CounterfactualSimulator().simulate(trace, CounterfactualToggle.VERIFICATION)
    assert result.baseline_failure_categories == []
    assert result.expected_improvement == 0.0


def test_result_never_mutates_the_original_trace():
    trace = ExecutionTrace(execution_id="e1", query="q", confidence=0.5, verification_trace={"claim_coverage": 0.0, "unsupported_claim_count": 9})
    CounterfactualSimulator().simulate(trace, CounterfactualToggle.VERIFICATION)
    assert trace.verification_trace == {"claim_coverage": 0.0, "unsupported_claim_count": 9}
