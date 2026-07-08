from app.graph_intelligence.propagation import GraphPropagationEngine, PropagationType
from app.ontology.labels import EntityLabel, RelationType
from app.reasoning.graph_models import ReasoningGraph, ReasoningNodeType
from app.repositories.graph_repository import InMemoryGraphRepository


def _chain_repo():
    """P-0 -> P-1 -> P-2 -> P-3, a simple chain so propagation distance is
    unambiguous to assert on."""
    repo = InMemoryGraphRepository()
    ids = [repo.create_node(EntityLabel.PUMP, f"P-{i}") for i in range(4)]
    for i in range(len(ids) - 1):
        repo.create_relation(ids[i], ids[i + 1], RelationType.CONNECTED_TO, confidence=0.5)
    return repo, ids


def test_propagate_reaches_direct_neighbor_at_depth_one():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE)

    step = next(s for s in result.steps if s.node_id == ids[1])
    assert step.depth == 1


def test_propagate_risk_decays_with_distance():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE, initial_risk=1.0, decay_factor=0.5, max_depth=3)

    risk_by_depth = {s.depth: s.propagated_risk for s in result.steps}
    assert risk_by_depth[1] > risk_by_depth[2] > risk_by_depth[3]


def test_propagate_respects_max_depth():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE, max_depth=1)

    assert all(s.depth <= 1 for s in result.steps)
    assert ids[1] in result.affected_node_ids
    assert ids[2] not in result.affected_node_ids


def test_propagate_stops_when_risk_falls_below_threshold():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE, initial_risk=0.1, decay_factor=0.1, max_depth=10)

    # 0.1 * 0.1 = 0.01 < threshold, so it should not even reach depth 1... but
    # depth-1 step itself is still recorded (computed before the threshold
    # check gates further expansion) — only depth 2+ should be absent.
    assert all(s.depth <= 1 for s in result.steps)


def test_propagate_does_not_revisit_the_source_node():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE)

    assert ids[0] not in result.affected_node_ids


def test_steps_are_sorted_by_risk_descending():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(ids[0], PropagationType.FAILURE)

    risks = [s.propagated_risk for s in result.steps]
    assert risks == sorted(risks, reverse=True)


def test_to_reasoning_nodes_produces_risk_typed_nodes():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)
    result = engine.propagate(ids[0], PropagationType.LEAK)

    nodes = engine.to_reasoning_nodes(result)

    assert all(n.node_type == ReasoningNodeType.RISK for n in nodes)
    assert all(n.confidence == step.propagated_risk for n, step in zip(nodes, result.steps, strict=True))


def test_merge_into_reasoning_graph_adds_nodes_and_edges():
    repo, ids = _chain_repo()
    engine = GraphPropagationEngine(repo)
    result = engine.propagate(ids[0], PropagationType.MAINTENANCE_DELAY)
    graph = ReasoningGraph(nodes=[], edges=[], summary="")

    engine.merge_into_reasoning_graph(graph, result)

    assert len(graph.nodes) == len(result.steps)
    assert len(graph.edges) == len(result.steps)
    assert all(e.source_id == ids[0] for e in graph.edges)


def test_propagate_handles_an_isolated_node_with_no_neighbors():
    repo = InMemoryGraphRepository()
    node_id = repo.create_node(EntityLabel.PUMP, "P-lonely")
    engine = GraphPropagationEngine(repo)

    result = engine.propagate(node_id, PropagationType.COMPLIANCE_VIOLATION)

    assert result.steps == []
