import threading
import time
from concurrent.futures import ThreadPoolExecutor

from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel
from app.planning.executor import TaskGraphExecutor
from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.retrieval.evidence import Evidence

_SLEEP_SECONDS = 0.05


def _entity(canonical_name: str) -> ExtractedEntity:
    return ExtractedEntity(
        id="e1",
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


class _SlowThreadSafeRetriever:
    def __init__(self, should_raise_for: frozenset[str] = frozenset()):
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self._should_raise_for = should_raise_for

    def retrieve(self, query: str, top_k: int = 8):
        time.sleep(_SLEEP_SECONDS)
        with self._lock:
            self.calls.append(query)
        if query in self._should_raise_for:
            raise RuntimeError(f"retriever exploded for {query!r}")
        return [Evidence(document_id="doc-1", page_number=1, chunk_id=f"text:{query}", text=f"text result for {query}", confidence=0.6, source="fusion")]


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def maintenance_history(self, canonical_name: str):
        return []

    def incident_history(self, canonical_name: str):
        return []

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        return []


def _five_independent_retrieval_nodes() -> TaskGraph:
    """Mirrors the real planner's shape: LOCATE_EQUIPMENT first, then five
    retrieval subtasks that all depend only on LOCATE_EQUIPMENT (not on each
    other) — exactly the two-level structure the real Planner produces."""
    return TaskGraph(
        nodes=[
            TaskNode(id="locate", type=TaskType.LOCATE_EQUIPMENT, description="locate"),
            TaskNode(id="maint", type=TaskType.RETRIEVE_MAINTENANCE_HISTORY, description="maint", dependencies=["locate"]),
            TaskNode(id="incidents", type=TaskType.RETRIEVE_INCIDENTS, description="incidents", dependencies=["locate"]),
            TaskNode(id="inspections", type=TaskType.RETRIEVE_INSPECTIONS, description="inspections", dependencies=["locate"]),
            TaskNode(id="standards", type=TaskType.RETRIEVE_STANDARDS, description="standards", dependencies=["locate"]),
            TaskNode(id="oem", type=TaskType.RETRIEVE_OEM_MANUAL, description="oem", dependencies=["locate"]),
        ]
    )


def test_no_executor_runs_the_five_independent_nodes_sequentially():
    retriever = _SlowThreadSafeRetriever()
    executor = TaskGraphExecutor(retriever, _FakeGraphRepository())
    graph = _five_independent_retrieval_nodes()

    start = time.perf_counter()
    executor.execute(graph, "why has P-101 failed", [_entity("P-101")])
    elapsed = time.perf_counter() - start

    assert elapsed >= _SLEEP_SECONDS * 4.5  # 5 sequential retrieve() calls (locate has no retrieve call)


def test_injected_executor_runs_the_five_independent_nodes_concurrently():
    retriever = _SlowThreadSafeRetriever()
    with ThreadPoolExecutor(max_workers=5) as pool:
        executor = TaskGraphExecutor(retriever, _FakeGraphRepository(), executor=pool)
        graph = _five_independent_retrieval_nodes()

        start = time.perf_counter()
        executor.execute(graph, "why has P-101 failed", [_entity("P-101")])
        elapsed = time.perf_counter() - start

    assert elapsed < _SLEEP_SECONDS * 3  # bounded by the slowest node in the level, not the sum of all five


def test_parallel_execution_respects_dependency_order_locate_runs_before_its_children():
    retriever = _SlowThreadSafeRetriever()
    with ThreadPoolExecutor(max_workers=5) as pool:
        executor = TaskGraphExecutor(retriever, _FakeGraphRepository(), executor=pool)
        graph = _five_independent_retrieval_nodes()

        executor.execute(graph, "why has P-101 failed", [_entity("P-101")])

    assert graph.get("locate").status == TaskStatus.COMPLETED
    for node_id in ("maint", "incidents", "inspections", "standards", "oem"):
        assert graph.get(node_id).status == TaskStatus.COMPLETED


def test_parallel_and_sequential_execution_produce_the_same_final_graph_state():
    graph_sequential = _five_independent_retrieval_nodes()
    TaskGraphExecutor(_SlowThreadSafeRetriever(), _FakeGraphRepository()).execute(
        graph_sequential, "why has P-101 failed", [_entity("P-101")]
    )

    graph_parallel = _five_independent_retrieval_nodes()
    with ThreadPoolExecutor(max_workers=5) as pool:
        TaskGraphExecutor(_SlowThreadSafeRetriever(), _FakeGraphRepository(), executor=pool).execute(
            graph_parallel, "why has P-101 failed", [_entity("P-101")]
        )

    for node_id in ("locate", "maint", "incidents", "inspections", "standards", "oem"):
        seq_node, par_node = graph_sequential.get(node_id), graph_parallel.get(node_id)
        assert seq_node.status == par_node.status
        assert [e.text for e in seq_node.result] == [e.text for e in par_node.result]


def test_a_single_node_failure_in_a_parallel_level_does_not_affect_sibling_nodes():
    retriever = _SlowThreadSafeRetriever(should_raise_for=frozenset({"why has P-101 failed incident failure"}))
    with ThreadPoolExecutor(max_workers=5) as pool:
        executor = TaskGraphExecutor(retriever, _FakeGraphRepository(), executor=pool)
        graph = _five_independent_retrieval_nodes()

        executor.execute(graph, "why has P-101 failed", [_entity("P-101")])

    assert graph.get("incidents").status == TaskStatus.FAILED
    assert graph.get("incidents").result == []
    for node_id in ("maint", "inspections", "standards", "oem"):
        assert graph.get(node_id).status == TaskStatus.COMPLETED
