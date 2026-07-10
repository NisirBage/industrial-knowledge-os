from app.claims.models import Claim, ClaimGraph, ClaimType
from app.evidence.models import EvidenceBundle
from app.execution.state_adapter import build_subsystem_traces
from app.faithfulness.models import FaithfulnessReport
from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.recommendation.verification import RecommendationVerification, RecommendationVerificationStatus, VerifiedRecommendationBundle
from app.reflection.models import ReflectionReport
from app.retrieval.evidence import Evidence


def test_build_subsystem_traces_from_empty_state_returns_empty_dict():
    assert build_subsystem_traces({}) == {}


def test_planner_trace_reflects_task_graph_node_and_failure_counts():
    task_graph = TaskGraph(nodes=[
        TaskNode(id="n1", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.FAILED),
        TaskNode(id="n2", type=TaskType.GENERAL_RETRIEVAL, description="d", status=TaskStatus.COMPLETED),
    ])
    traces = build_subsystem_traces({"task_graph": task_graph})
    assert traces["planner"] == {"node_count": 2, "failed_node_count": 1, "task_type": "general_retrieval"}


def test_retrieval_trace_reflects_evidence_bundle_signals():
    bundle = EvidenceBundle(
        evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="t")],
        graph_evidence=[], timeline=[], metadata=[], missing_information=["gap"], confidence=0.6,
    )
    traces = build_subsystem_traces({"evidence_bundle": bundle})
    assert traces["retrieval"] == {"evidence_count": 1, "confidence": 0.6, "missing_information_count": 1}


def test_claim_extraction_trace_reflects_claim_count():
    claim_graph = ClaimGraph(claims=[Claim(id="c1", claim_type=ClaimType.FACT, text="t", subject="s", predicate="p")], relations=[])
    traces = build_subsystem_traces({"claim_graph": claim_graph})
    assert traces["claim_extraction"] == {"claim_count": 1}


def test_verification_nli_and_faithfulness_traces_derive_from_faithfulness_report():
    report = FaithfulnessReport(
        overall_faithfulness=0.7, sentence_faithfulness=[], recommendation_faithfulness=1.0, graph_faithfulness=1.0,
        claim_coverage=0.5, unsupported_claim_count=2, hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=0.3,
    )
    traces = build_subsystem_traces({"faithfulness_report": report})
    assert traces["verification"] == {"claim_coverage": 0.5, "unsupported_claim_count": 2}
    assert traces["nli"] == {"contradiction_rate": 0.3}
    assert traces["faithfulness"] == {"overall_faithfulness": 0.7}


def test_recommendation_trace_reflects_verification_counts():
    verification = VerifiedRecommendationBundle(
        verified=["r1"], rejected=["r2"],
        verifications={
            "r1": RecommendationVerification("r1", RecommendationVerificationStatus.VERIFIED),
            "r2": RecommendationVerification("r2", RecommendationVerificationStatus.REJECTED),
        },
        overall_confidence=0.5,
    )
    traces = build_subsystem_traces({"recommendation_verification": verification})
    assert traces["recommendation"] == {"verified_count": 1, "rejected_count": 1}


def test_reflection_trace_reflects_reflection_report():
    report = ReflectionReport(
        evidence_sufficient=True, evidence_sufficiency_note="n", confidence_justified=True,
        confidence_justification_note="n", recommendations_verified_ratio=0.8,
        additional_retrieval_suggested=False, additional_retrieval_note="n", summary="s",
    )
    traces = build_subsystem_traces({"reflection_report": report})
    assert traces["reflection"] == {
        "evidence_sufficient": True, "confidence_justified": True, "recommendations_verified_ratio": 0.8,
    }
