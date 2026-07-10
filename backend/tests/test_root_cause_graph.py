from app.error_analysis.models import FailureSeverity
from app.execution.models import ExecutionEvent, ExecutionTrace
from app.failure_intelligence.models import FailurePattern
from app.failure_intelligence.root_cause_graph import (
    RootCauseEdge,
    RootCauseEdgeType,
    RootCauseGraph,
    RootCauseGraphBuilder,
    RootCauseNode,
    RootCauseNodeType,
)


def _event(component: str, event_type: str) -> ExecutionEvent:
    return ExecutionEvent(event_type=event_type, component=component, timestamp="2026-01-01T00:00:00+00:00", duration=0.1)


def _failing_trace() -> ExecutionTrace:
    events = [_event("planning", "planning_started"), _event("planning", "planning_finished"), _event("retrieval", "retrieval_started"), _event("retrieval", "retrieval_finished")]
    return ExecutionTrace(execution_id="e1", query="Why did P-101 fail?", events=events, confidence=0.2, retrieval_trace=None)


def test_build_creates_execution_and_component_nodes():
    graph = RootCauseGraphBuilder().build(_failing_trace())

    node_ids = {n.id for n in graph.nodes}
    assert "execution:e1" in node_ids
    assert "component:planning" in node_ids
    assert "component:retrieval" in node_ids


def test_build_creates_a_failure_node_with_caused_by_edge_to_primary_component():
    graph = RootCauseGraphBuilder().build(_failing_trace())

    failure_node = next(n for n in graph.nodes if n.type == RootCauseNodeType.FAILURE)
    assert failure_node.label == "retrieval_failure"

    caused_by = [e for e in graph.edges if e.edge_type == RootCauseEdgeType.CAUSED_BY]
    assert len(caused_by) == 1
    assert caused_by[0].source_id == failure_node.id
    target = graph.node(caused_by[0].target_id)
    assert target.label == "HybridRetriever"


def test_dominant_cause_returns_the_caused_by_target():
    graph = RootCauseGraphBuilder().build(_failing_trace())
    failure_node = next(n for n in graph.nodes if n.type == RootCauseNodeType.FAILURE)

    dominant = graph.dominant_cause(failure_node.id)
    assert dominant.label == "HybridRetriever"


def test_dominant_cause_returns_none_when_no_caused_by_edge():
    graph = RootCauseGraph(nodes=[RootCauseNode("f1", RootCauseNodeType.FAILURE, "x")], edges=[])
    assert graph.dominant_cause("f1") is None


def test_traverse_visits_reachable_nodes_via_bfs():
    graph = RootCauseGraphBuilder().build(_failing_trace())
    reachable = graph.traverse("execution:e1")
    reachable_ids = {n.id for n in reachable}
    assert "execution:e1" in reachable_ids
    assert any(n.type == RootCauseNodeType.FAILURE for n in reachable)
    assert any(n.type == RootCauseNodeType.COMPONENT for n in reachable)


def test_dependency_depth_counts_causal_chain_length():
    graph = RootCauseGraph(
        nodes=[
            RootCauseNode("f1", RootCauseNodeType.FAILURE, "f1"),
            RootCauseNode("c1", RootCauseNodeType.COMPONENT, "c1"),
            RootCauseNode("c2", RootCauseNodeType.COMPONENT, "c2"),
        ],
        edges=[
            RootCauseEdge("f1", "c1", RootCauseEdgeType.CAUSED_BY),
            RootCauseEdge("c1", "c2", RootCauseEdgeType.CONTRIBUTED_TO),
        ],
    )
    assert graph.dependency_depth("f1") == 2
    assert graph.dependency_depth("c2") == 0


def test_critical_path_finds_shortest_route():
    graph = RootCauseGraph(
        nodes=[RootCauseNode(nid, RootCauseNodeType.COMPONENT, nid) for nid in ["a", "b", "c"]],
        edges=[RootCauseEdge("a", "b", RootCauseEdgeType.CORRELATED_WITH), RootCauseEdge("b", "c", RootCauseEdgeType.CORRELATED_WITH)],
    )
    path = graph.critical_path("a", "c")
    assert [n.id for n in path] == ["a", "b", "c"]


def test_critical_path_returns_none_when_unreachable():
    graph = RootCauseGraph(nodes=[RootCauseNode("a", RootCauseNodeType.COMPONENT, "a"), RootCauseNode("b", RootCauseNodeType.COMPONENT, "b")], edges=[])
    assert graph.critical_path("a", "b") is None


def test_to_export_dict_round_trips_nodes_and_edges():
    graph = RootCauseGraphBuilder().build(_failing_trace())
    exported = graph.to_export_dict()
    assert len(exported["nodes"]) == len(graph.nodes)
    assert len(exported["edges"]) == len(graph.edges)
    assert all("id" in n and "type" in n for n in exported["nodes"])


def test_build_attaches_matching_pattern_node():
    pattern = FailurePattern(
        pattern_id="p1", category="retrieval_failure", support=3, frequency=0.5,
        severity=FailureSeverity.HIGH, affected_components=["HybridRetriever"],
        example_execution_ids=["e1"], confidence=0.7, description="d",
    )
    graph = RootCauseGraphBuilder().build(_failing_trace(), patterns=[pattern])

    pattern_node = next(n for n in graph.nodes if n.type == RootCauseNodeType.PATTERN)
    assert pattern_node.id == "pattern:p1"
    assert any(e.source_id == pattern_node.id and e.edge_type == RootCauseEdgeType.CORRELATED_WITH for e in graph.edges)


def test_build_creates_evidence_node_when_retrieval_trace_present():
    trace = ExecutionTrace(
        execution_id="e2", query="q", events=[_event("retrieval", "retrieval_started"), _event("retrieval", "retrieval_finished")],
        confidence=0.8, retrieval_trace={"evidence_count": 3, "confidence": 0.8, "missing_information_count": 0},
    )
    graph = RootCauseGraphBuilder().build(trace)
    evidence_node = next((n for n in graph.nodes if n.type == RootCauseNodeType.EVIDENCE), None)
    assert evidence_node is not None
    assert "3 evidence item(s)" == evidence_node.label
