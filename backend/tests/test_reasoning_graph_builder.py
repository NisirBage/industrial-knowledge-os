from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle
from app.reasoning.graph_models import ReasoningNodeType
from app.reasoning.reasoning_graph_builder import ReasoningGraphBuilder
from app.retrieval.evidence import Evidence


def _bundle(evidence=None, graph_evidence=None, missing_information=None, confidence=0.7) -> EvidenceBundle:
    return EvidenceBundle(
        evidence=evidence or [],
        graph_evidence=graph_evidence or [],
        timeline=[],
        metadata=[],
        missing_information=missing_information or [],
        confidence=confidence,
    )


def test_every_text_evidence_item_becomes_a_fact_node():
    bundle = _bundle(evidence=[Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 needs lubrication.", confidence=0.7)])

    graph = ReasoningGraphBuilder().build(bundle, [])

    facts = graph.nodes_of_type(ReasoningNodeType.FACT)
    assert len(facts) == 1
    assert facts[0].supporting_evidence_ids == ["c1"]


def test_causal_relation_triple_produces_cause_effect_and_risk_nodes():
    graph_evidence = [
        Evidence(document_id="", page_number=0, chunk_id="g1", text="P-101 FAILED_DUE_TO Bearing Wear", source="graph", confidence=0.6)
    ]
    bundle = _bundle(graph_evidence=graph_evidence)

    graph = ReasoningGraphBuilder().build(bundle, [])

    assert len(graph.nodes_of_type(ReasoningNodeType.CAUSE)) == 1
    assert len(graph.nodes_of_type(ReasoningNodeType.EFFECT)) == 1
    assert len(graph.nodes_of_type(ReasoningNodeType.RISK)) == 1
    assert any(e.relationship == "causes" for e in graph.edges)
    assert any(e.relationship == "indicates_risk" for e in graph.edges)


def test_non_causal_relation_triple_does_not_produce_cause_effect_nodes():
    graph_evidence = [
        Evidence(document_id="", page_number=0, chunk_id="g1", text="P-101 CONNECTED_TO V-2201A", source="graph", confidence=0.6)
    ]
    bundle = _bundle(graph_evidence=graph_evidence)

    graph = ReasoningGraphBuilder().build(bundle, [])

    assert graph.nodes_of_type(ReasoningNodeType.CAUSE) == []
    assert graph.nodes_of_type(ReasoningNodeType.RISK) == []
    assert len(graph.nodes_of_type(ReasoningNodeType.FACT)) == 1


def test_contradiction_produces_hypothesis_node():
    contradiction = Contradiction(
        conflicting_evidence=["c1", "c2"],
        confidence=0.6,
        severity=Severity.MEDIUM,
        resolution_strategy="verify",
        recommended_clarification="Confirm valve state",
    )

    graph = ReasoningGraphBuilder().build(_bundle(), [contradiction])

    hypotheses = graph.nodes_of_type(ReasoningNodeType.HYPOTHESIS)
    assert len(hypotheses) == 1
    assert hypotheses[0].confidence == round(1 - 0.6, 2)
    assert hypotheses[0].supporting_evidence_ids == ["c1", "c2"]


def test_missing_information_produces_missing_evidence_nodes():
    bundle = _bundle(missing_information=["No inspection records found"])

    graph = ReasoningGraphBuilder().build(bundle, [])

    missing_nodes = graph.nodes_of_type(ReasoningNodeType.MISSING_EVIDENCE)
    assert len(missing_nodes) == 1
    assert missing_nodes[0].description == "No inspection records found"


def test_empty_bundle_produces_unknown_node():
    graph = ReasoningGraphBuilder().build(_bundle(confidence=0.0), [])
    assert len(graph.nodes_of_type(ReasoningNodeType.UNKNOWN)) == 1


def test_summary_is_deterministic_for_same_input():
    bundle = _bundle(evidence=[Evidence(document_id="d1", page_number=1, chunk_id="c1", text="fact", confidence=0.5)])
    summary_a = ReasoningGraphBuilder().build(bundle, []).summary
    summary_b = ReasoningGraphBuilder().build(bundle, []).summary
    assert summary_a == summary_b
    assert "1 piece(s) of evidence" in summary_a
