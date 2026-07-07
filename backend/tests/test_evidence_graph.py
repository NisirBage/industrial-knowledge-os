from app.evidence.graph import EvidenceGraphBuilder, EvidenceNodeType
from app.evidence.models import EvidenceBundle
from app.retrieval.evidence import Evidence


def test_build_creates_question_node_and_evidence_nodes_with_edges():
    bundle = EvidenceBundle(
        evidence=[Evidence(document_id="doc-1", page_number=1, chunk_id="c1", text="Pump P-101 requires lubrication.", confidence=0.7)],
        graph_evidence=[],
        timeline=[],
        metadata=[],
        missing_information=[],
        confidence=0.7,
    )

    graph = EvidenceGraphBuilder().build("How often is P-101 lubricated?", bundle)

    question_nodes = [n for n in graph.nodes if n.node_type == EvidenceNodeType.QUESTION]
    evidence_nodes = [n for n in graph.nodes if n.node_type == EvidenceNodeType.EVIDENCE]
    assert len(question_nodes) == 1
    assert len(evidence_nodes) == 1
    assert any(e.source_id == "question" and e.target_id == "evidence:c1" for e in graph.edges)


def test_build_creates_relationship_and_entity_nodes_from_graph_evidence():
    bundle = EvidenceBundle(
        evidence=[],
        graph_evidence=[
            Evidence(
                document_id="",
                page_number=0,
                chunk_id="g1",
                text="P-101 CONNECTED_TO V-2201A",
                entity_ids=["entity-p101", "entity-v2201a"],
                source="graph",
                confidence=0.55,
            )
        ],
        timeline=[],
        metadata=[],
        missing_information=[],
        confidence=0.55,
    )

    graph = EvidenceGraphBuilder().build("What is P-101 connected to?", bundle)

    relationship_nodes = [n for n in graph.nodes if n.node_type == EvidenceNodeType.RELATIONSHIP]
    entity_nodes = [n for n in graph.nodes if n.node_type == EvidenceNodeType.ENTITY]
    assert len(relationship_nodes) == 1
    assert {n.entity_id for n in entity_nodes} == {"entity-p101", "entity-v2201a"}
    assert any(e.source_id == "relationship:g1" and e.target_id == "entity:entity-p101" for e in graph.edges)


def test_build_deduplicates_shared_entities_across_multiple_graph_evidence_items():
    shared_evidence = [
        Evidence(document_id="", page_number=0, chunk_id="g1", text="A REL B", entity_ids=["e1", "e2"], source="graph", confidence=0.5),
        Evidence(document_id="", page_number=0, chunk_id="g2", text="A REL C", entity_ids=["e1", "e3"], source="graph", confidence=0.5),
    ]
    bundle = EvidenceBundle(evidence=[], graph_evidence=shared_evidence, timeline=[], metadata=[], missing_information=[], confidence=0.5)

    graph = EvidenceGraphBuilder().build("query", bundle)

    entity_node_ids = [n.id for n in graph.nodes if n.node_type == EvidenceNodeType.ENTITY]
    assert entity_node_ids.count("entity:e1") == 1
