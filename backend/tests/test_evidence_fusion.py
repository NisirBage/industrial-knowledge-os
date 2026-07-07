from app.evidence.fusion import EvidenceFusionEngine
from app.planning.models import TaskGraph, TaskNode, TaskType
from app.retrieval.evidence import Evidence


def _evidence(chunk_id: str, text: str, confidence: float, source: str = "fusion", document_id: str = "doc-1") -> Evidence:
    return Evidence(document_id=document_id, page_number=1, chunk_id=chunk_id, text=text, confidence=confidence, source=source)


def test_fuse_deduplicates_by_chunk_id_keeping_highest_confidence():
    node = TaskNode(id="a", type=TaskType.GENERAL_RETRIEVAL, description="a")
    node.result = [_evidence("c1", "low", 0.3), _evidence("c1", "high", 0.9)]
    graph = TaskGraph(nodes=[node])

    bundle = EvidenceFusionEngine().fuse(graph, [])

    assert len(bundle.evidence) == 1
    assert bundle.evidence[0].confidence == 0.9
    assert bundle.evidence[0].text == "high"


def test_fuse_ranks_evidence_by_confidence_descending():
    node = TaskNode(id="a", type=TaskType.GENERAL_RETRIEVAL, description="a")
    node.result = [_evidence("c1", "low", 0.2), _evidence("c2", "high", 0.8)]
    graph = TaskGraph(nodes=[node])

    bundle = EvidenceFusionEngine().fuse(graph, [])

    assert [e.chunk_id for e in bundle.evidence] == ["c2", "c1"]


def test_fuse_separates_graph_evidence_from_text_evidence():
    node = TaskNode(id="a", type=TaskType.GENERAL_RETRIEVAL, description="a")
    node.result = [_evidence("c1", "text", 0.5)]
    graph = TaskGraph(nodes=[node])
    graph_evidence = [_evidence("g1", "P-101 CONNECTED_TO V-2201A", 0.55, source="graph")]

    bundle = EvidenceFusionEngine().fuse(graph, graph_evidence)

    assert len(bundle.evidence) == 1
    assert len(bundle.graph_evidence) == 1
    assert bundle.graph_evidence[0].chunk_id == "g1"


def test_fuse_builds_chronological_timeline_from_dates_in_text():
    node = TaskNode(id="a", type=TaskType.GENERAL_RETRIEVAL, description="a")
    node.result = [
        _evidence("c1", "Failure occurred in March 2024.", 0.6),
        _evidence("c2", "Inspection completed in January 2025.", 0.6),
        _evidence("c3", "No date mentioned here.", 0.6),
    ]
    graph = TaskGraph(nodes=[node])

    bundle = EvidenceFusionEngine().fuse(graph, [])

    dated = [event for event in bundle.timeline if event.timestamp is not None]
    assert [event.timestamp for event in dated] == ["March 2024", "January 2025"]
    assert bundle.timeline[-1].timestamp is None


def test_fuse_reports_missing_information_for_empty_task_nodes():
    empty_node = TaskNode(id="a", type=TaskType.RETRIEVE_STANDARDS, description="Retrieve applicable standards")
    graph = TaskGraph(nodes=[empty_node])

    bundle = EvidenceFusionEngine().fuse(graph, [])

    assert bundle.missing_information == ["Retrieve applicable standards"]
    assert bundle.confidence == 0.0


def test_fuse_confidence_reflects_evidence_quality_and_missing_info_penalty():
    node = TaskNode(id="a", type=TaskType.GENERAL_RETRIEVAL, description="a")
    node.result = [_evidence("c1", "strong evidence", 0.9)]
    empty_node = TaskNode(id="b", type=TaskType.RETRIEVE_STANDARDS, description="Retrieve standards")
    graph = TaskGraph(nodes=[node, empty_node])

    bundle = EvidenceFusionEngine().fuse(graph, [])

    assert 0.0 < bundle.confidence < 0.9
    assert bundle.missing_information == ["Retrieve standards"]
