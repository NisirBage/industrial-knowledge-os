"""Adapts the orchestrator's `AgentState` into the small per-subsystem trace
summaries `ExecutionTrace` carries (O3.3 Component 10 migration support).

Each summary is a handful of real numbers already computed by that stage —
never a duplicate of the full domain object (that would be exactly the
"duplicated telemetry" this milestone exists to eliminate). Subsystems not
invoked inside live chat (Compression, Vision, Prediction, Graph
Intelligence) are simply absent from the returned dict, matching
`ExecutionTrace`'s `None`-by-default convention for un-invoked slots.
"""

from app.orchestrator.state import AgentState
from app.planning.models import TaskStatus


def build_subsystem_traces(state: AgentState) -> dict[str, dict]:
    traces: dict[str, dict] = {}

    task_graph = state.get("task_graph")
    if task_graph is not None:
        failed = [n for n in task_graph.nodes if n.status == TaskStatus.FAILED]
        traces["planner"] = {
            "node_count": len(task_graph.nodes),
            "failed_node_count": len(failed),
            # O3.4 Component 2 (Failure Clustering — intent-based): the
            # first task node's type is the same real query-intent
            # classification the Planner already computed; reused as-is,
            # not re-derived.
            "task_type": task_graph.nodes[0].type.value if task_graph.nodes else None,
        }

    evidence_bundle = state.get("evidence_bundle")
    if evidence_bundle is not None:
        traces["retrieval"] = {
            "evidence_count": len(evidence_bundle.all_evidence()),
            "confidence": evidence_bundle.confidence,
            "missing_information_count": len(evidence_bundle.missing_information),
        }

    claim_graph = state.get("claim_graph")
    if claim_graph is not None:
        traces["claim_extraction"] = {"claim_count": len(claim_graph.claims)}

    faithfulness_report = state.get("faithfulness_report")
    if faithfulness_report is not None:
        traces["verification"] = {
            "claim_coverage": faithfulness_report.claim_coverage,
            "unsupported_claim_count": faithfulness_report.unsupported_claim_count,
        }
        traces["nli"] = {"contradiction_rate": faithfulness_report.contradiction_rate}
        traces["faithfulness"] = {"overall_faithfulness": faithfulness_report.overall_faithfulness}

    recommendation_verification = state.get("recommendation_verification")
    if recommendation_verification is not None:
        traces["recommendation"] = {
            "verified_count": len(recommendation_verification.verified),
            "rejected_count": len(recommendation_verification.rejected),
        }

    reflection_report = state.get("reflection_report")
    if reflection_report is not None:
        traces["reflection"] = {
            "evidence_sufficient": reflection_report.evidence_sufficient,
            "confidence_justified": reflection_report.confidence_justified,
            "recommendations_verified_ratio": reflection_report.recommendations_verified_ratio,
        }

    return traces
