"""GraphPropagationEngine (M9 Component 7): simulates how a risk (equipment
failure, leak, maintenance delay, inspection gap, compliance violation)
propagates outward across the graph from its source node — a real,
deterministic BFS with exponential distance-decay, not a black-box model.

Integrates with the existing Industrial Reasoning Graph (M3) by emitting
ordinary `ReasoningNode(node_type=RISK, ...)` entries — a type the
Reasoning Graph's model already declared — so a propagation result can be
folded directly into a `ReasoningGraph` a Chat query already produces,
rather than living in a separate, disconnected risk report.
"""

from dataclasses import dataclass
from enum import Enum

from app.reasoning.graph_models import ReasoningEdge, ReasoningGraph, ReasoningNode, ReasoningNodeType
from app.repositories.graph_repository import GraphRepositoryProtocol

_DEFAULT_MAX_DEPTH = 3
_DEFAULT_DECAY_FACTOR = 0.5
_MIN_RISK_THRESHOLD = 0.05


class PropagationType(str, Enum):
    FAILURE = "failure"
    LEAK = "leak"
    MAINTENANCE_DELAY = "maintenance_delay"
    INSPECTION_GAP = "inspection_gap"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass(frozen=True)
class PropagationStep:
    node_id: str
    canonical_name: str
    depth: int
    propagated_risk: float
    path: list[str]


@dataclass(frozen=True)
class PropagationResult:
    source_node_id: str
    propagation_type: PropagationType
    steps: list[PropagationStep]

    @property
    def affected_node_ids(self) -> list[str]:
        return [step.node_id for step in self.steps]


class GraphPropagationEngine:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def propagate(
        self,
        source_node_id: str,
        propagation_type: PropagationType,
        initial_risk: float = 1.0,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        decay_factor: float = _DEFAULT_DECAY_FACTOR,
    ) -> PropagationResult:
        # (node_id, canonical_name, depth, risk, path) — canonical_name is
        # carried forward from the neighbor dict that discovered this node
        # (find_neighbors already returns it), rather than looked up again;
        # GraphRepositoryProtocol has no "get node by id" reverse lookup.
        visited: dict[str, PropagationStep] = {}
        frontier: list[tuple[str, str, int, float, list[str]]] = [(source_node_id, source_node_id, 0, initial_risk, [source_node_id])]
        visited_ids = {source_node_id}

        while frontier:
            node_id, canonical_name, depth, risk, path = frontier.pop(0)
            if depth > 0:
                visited[node_id] = PropagationStep(node_id=node_id, canonical_name=canonical_name, depth=depth, propagated_risk=round(risk, 4), path=path)

            if depth >= max_depth or risk < _MIN_RISK_THRESHOLD:
                continue

            for neighbor in self._graph_repository.find_neighbors(node_id, depth=1):
                if neighbor["id"] in visited_ids:
                    continue
                visited_ids.add(neighbor["id"])
                next_risk = risk * decay_factor
                frontier.append((neighbor["id"], neighbor["canonical_name"], depth + 1, next_risk, path + [neighbor["id"]]))

        steps = sorted(visited.values(), key=lambda s: (-s.propagated_risk, s.depth))
        return PropagationResult(source_node_id=source_node_id, propagation_type=propagation_type, steps=steps)

    @staticmethod
    def to_reasoning_nodes(result: PropagationResult) -> list[ReasoningNode]:
        return [
            ReasoningNode(
                id=f"risk-propagation-{result.propagation_type.value}-{step.node_id}",
                node_type=ReasoningNodeType.RISK,
                description=(
                    f"{result.propagation_type.value.replace('_', ' ').title()} risk propagated from "
                    f"{result.source_node_id} to {step.canonical_name} (depth {step.depth}, path: {' -> '.join(step.path)})"
                ),
                confidence=step.propagated_risk,
                supporting_evidence_ids=step.path,
            )
            for step in result.steps
        ]

    @staticmethod
    def merge_into_reasoning_graph(graph: ReasoningGraph, result: PropagationResult) -> None:
        risk_nodes = GraphPropagationEngine.to_reasoning_nodes(result)
        for node in risk_nodes:
            graph.add_node(node)
            graph.add_edge(ReasoningEdge(source_id=result.source_node_id, target_id=node.id, relationship="propagates_to"))
