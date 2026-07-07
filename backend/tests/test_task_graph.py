import pytest

from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.retrieval.evidence import Evidence


def _node(node_id: str, deps: list[str] | None = None) -> TaskNode:
    return TaskNode(id=node_id, type=TaskType.GENERAL_RETRIEVAL, description=node_id, dependencies=deps or [])


def test_topological_order_respects_dependencies():
    graph = TaskGraph(nodes=[_node("c", ["a", "b"]), _node("a"), _node("b", ["a"])])

    order_ids = [n.id for n in graph.topological_order()]

    assert order_ids.index("a") < order_ids.index("b")
    assert order_ids.index("b") < order_ids.index("c")


def test_topological_order_raises_on_cycle():
    graph = TaskGraph(nodes=[_node("a", ["b"]), _node("b", ["a"])])

    with pytest.raises(ValueError):
        graph.topological_order()


def test_get_raises_key_error_for_unknown_id():
    graph = TaskGraph(nodes=[_node("a")])
    with pytest.raises(KeyError):
        graph.get("missing")


def test_all_completed_false_until_every_node_completed():
    graph = TaskGraph(nodes=[_node("a"), _node("b")])
    assert not graph.all_completed()

    graph.get("a").status = TaskStatus.COMPLETED
    assert not graph.all_completed()

    graph.get("b").status = TaskStatus.COMPLETED
    assert graph.all_completed()


def test_flattened_evidence_combines_all_node_results():
    graph = TaskGraph(nodes=[_node("a"), _node("b")])
    evidence_a = Evidence(document_id="d1", page_number=1, chunk_id="c1", text="a")
    evidence_b = Evidence(document_id="d1", page_number=1, chunk_id="c2", text="b")
    graph.get("a").result = [evidence_a]
    graph.get("b").result = [evidence_b]

    assert graph.flattened_evidence() == [evidence_a, evidence_b]
