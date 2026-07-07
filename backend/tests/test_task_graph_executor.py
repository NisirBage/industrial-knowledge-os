import uuid

from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel
from app.planning.executor import TaskGraphExecutor
from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.retrieval.evidence import Evidence


def _entity(canonical_name: str) -> ExtractedEntity:
    return ExtractedEntity(
        id=str(uuid.uuid4()),
        label=EntityLabel.PUMP,
        canonical_name=canonical_name,
        aliases=frozenset({canonical_name}),
        confidence=0.9,
        document_id="query",
        chunk_id="query",
        page_number=0,
        start_char=0,
        end_char=len(canonical_name),
    )


class _FakeRetriever:
    def __init__(self, should_raise: bool = False):
        self.calls: list[str] = []
        self._should_raise = should_raise

    def retrieve(self, query: str, top_k: int = 8):
        if self._should_raise:
            raise RuntimeError("retriever exploded")
        self.calls.append(query)
        return [Evidence(document_id="doc-1", page_number=1, chunk_id=f"text:{len(self.calls)}", text=f"text result for {query}", confidence=0.6, source="fusion")]


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def maintenance_history(self, canonical_name: str):
        return [{"id": "node-hist1", "type": "Inspection", "canonical_name": "Inspection Jan 2025", "relation_type": "INSPECTED_BY"}]

    def incident_history(self, canonical_name: str):
        return [{"id": "node-incident1", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        return [{"id": "node-std1", "type": "Standard", "canonical_name": "ISO 9001", "relation_type": "USES_STANDARD"}]


def test_locate_equipment_produces_graph_evidence_when_entity_found():
    executor = TaskGraphExecutor(_FakeRetriever(), _FakeGraphRepository())
    graph = TaskGraph(nodes=[TaskNode(id="locate", type=TaskType.LOCATE_EQUIPMENT, description="locate")])

    executor.execute(graph, "why has P-101 failed", [_entity("P-101")])

    node = graph.get("locate")
    assert node.status == TaskStatus.COMPLETED
    assert len(node.result) == 1
    assert node.result[0].source == "graph"


def test_locate_equipment_empty_result_when_entity_not_in_graph():
    executor = TaskGraphExecutor(_FakeRetriever(), _FakeGraphRepository())
    graph = TaskGraph(nodes=[TaskNode(id="locate", type=TaskType.LOCATE_EQUIPMENT, description="locate")])

    executor.execute(graph, "why has X-999 failed", [_entity("X-999")])

    node = graph.get("locate")
    assert node.status == TaskStatus.COMPLETED
    assert node.result == []


def test_maintenance_history_combines_graph_and_text_evidence():
    retriever = _FakeRetriever()
    executor = TaskGraphExecutor(retriever, _FakeGraphRepository())
    graph = TaskGraph(nodes=[TaskNode(id="maint", type=TaskType.RETRIEVE_MAINTENANCE_HISTORY, description="maint history")])

    executor.execute(graph, "why has P-101 failed", [_entity("P-101")])

    node = graph.get("maint")
    sources = {e.source for e in node.result}
    assert sources == {"graph", "fusion"}
    assert retriever.calls  # text retrieval was actually invoked


def test_general_retrieval_calls_retriever_with_plain_query():
    retriever = _FakeRetriever()
    executor = TaskGraphExecutor(retriever, _FakeGraphRepository())
    graph = TaskGraph(nodes=[TaskNode(id="general", type=TaskType.GENERAL_RETRIEVAL, description="general")])

    executor.execute(graph, "what is the safety policy", [])

    assert retriever.calls == ["what is the safety policy"]
    assert graph.get("general").status == TaskStatus.COMPLETED


def test_node_marked_failed_on_exception():
    executor = TaskGraphExecutor(_FakeRetriever(should_raise=True), _FakeGraphRepository())
    graph = TaskGraph(nodes=[TaskNode(id="general", type=TaskType.GENERAL_RETRIEVAL, description="general")])

    executor.execute(graph, "anything", [])

    node = graph.get("general")
    assert node.status == TaskStatus.FAILED
    assert node.result == []
    assert node.confidence == 0.0
