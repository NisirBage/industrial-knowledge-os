"""Root Cause Graph (O3.4 Component 3).

Nodes: Failure / Execution / Claim / Recommendation / Component / Evidence /
Pattern. Edges: CAUSED_BY / CONTRIBUTED_TO / CORRELATED_WITH / PREVENTED_BY /
AMPLIFIED_BY.

`RootCauseGraphBuilder.build()` constructs the graph entirely from one real
`ExecutionTrace` plus the `DetectedFailure`s `ErrorAnalyzer.analyze_trace()`
already found for it (O3.3 Component 10) — no independent failure detection,
no duplicated telemetry. Claim/Recommendation/Evidence nodes are aggregate
per-execution stand-ins (one node summarizing that stage's trace dict),
since `ExecutionTrace` deliberately doesn't carry individual claim/evidence
ids (that would duplicate the Claim Graph/Evidence Bundle themselves).

`PREVENTED_BY` is a valid edge type any producer can emit given a real
preventative signal (e.g. a Counterfactual Simulator result showing a
specific safeguard stopped a failure) — the default builder here doesn't
emit it, since ExecutionTrace alone carries no such signal.

`AMPLIFIED_BY` edges the builder does emit are an explicitly documented
correlation heuristic (upstream retrieval/ranking weakness often coincides
with downstream verification/faithfulness/recommendation problems) — never
asserted as proven causality.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from app.error_analysis.analyzer import ErrorAnalyzer
from app.error_analysis.models import DetectedFailure
from app.execution.models import ExecutionTrace
from app.failure_intelligence.models import FailurePattern

_UPSTREAM_AMPLIFIER_CATEGORIES = {"retrieval_failure", "ranking_failure"}
_DOWNSTREAM_AMPLIFIED_CATEGORIES = {"verification_failure", "faithfulness_failure", "recommendation_failure", "nli_failure"}


class RootCauseNodeType(str, Enum):
    FAILURE = "failure"
    EXECUTION = "execution"
    CLAIM = "claim"
    RECOMMENDATION = "recommendation"
    COMPONENT = "component"
    EVIDENCE = "evidence"
    PATTERN = "pattern"


class RootCauseEdgeType(str, Enum):
    CAUSED_BY = "caused_by"
    CONTRIBUTED_TO = "contributed_to"
    CORRELATED_WITH = "correlated_with"
    PREVENTED_BY = "prevented_by"
    AMPLIFIED_BY = "amplified_by"


@dataclass(frozen=True)
class RootCauseNode:
    id: str
    type: RootCauseNodeType
    label: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RootCauseEdge:
    source_id: str
    target_id: str
    edge_type: RootCauseEdgeType
    confidence: float = 1.0
    evidence: str = ""


@dataclass(frozen=True)
class RootCauseGraph:
    nodes: list[RootCauseNode] = field(default_factory=list)
    edges: list[RootCauseEdge] = field(default_factory=list)

    def node(self, node_id: str) -> RootCauseNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def outgoing_edges(self, node_id: str) -> list[RootCauseEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def incoming_edges(self, node_id: str) -> list[RootCauseEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def traverse(self, from_id: str) -> list[RootCauseNode]:
        """BFS forward traversal along every outgoing edge."""
        visited: set[str] = set()
        order: list[RootCauseNode] = []
        queue: deque[str] = deque([from_id])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            node = self.node(current)
            if node is not None:
                order.append(node)
            for edge in self.outgoing_edges(current):
                if edge.target_id not in visited:
                    queue.append(edge.target_id)
        return order

    def dominant_cause(self, failure_node_id: str) -> RootCauseNode | None:
        """The highest-confidence CAUSED_BY target for this failure node.
        Never fabricated: returns None if no CAUSED_BY edge exists."""
        caused_by_edges = [e for e in self.outgoing_edges(failure_node_id) if e.edge_type == RootCauseEdgeType.CAUSED_BY]
        if not caused_by_edges:
            return None
        best = max(caused_by_edges, key=lambda e: e.confidence)
        return self.node(best.target_id)

    def dependency_depth(self, node_id: str) -> int:
        """Longest CAUSED_BY/CONTRIBUTED_TO chain reachable from this node."""

        def _depth(current_id: str, visited: frozenset[str]) -> int:
            if current_id in visited:
                return 0
            visited = visited | {current_id}
            causal_edges = [
                e for e in self.outgoing_edges(current_id)
                if e.edge_type in (RootCauseEdgeType.CAUSED_BY, RootCauseEdgeType.CONTRIBUTED_TO)
            ]
            if not causal_edges:
                return 0
            return 1 + max(_depth(e.target_id, visited) for e in causal_edges)

        return _depth(node_id, frozenset())

    def critical_path(self, from_id: str, to_id: str) -> list[RootCauseNode] | None:
        """Shortest path (BFS, any edge type) from `from_id` to `to_id`, or
        None if unreachable."""
        if from_id == to_id:
            node = self.node(from_id)
            return [node] if node else None

        visited = {from_id}
        queue: deque[list[str]] = deque([[from_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            for edge in self.outgoing_edges(current):
                if edge.target_id == to_id:
                    return [self.node(nid) for nid in (*path, to_id)]
                if edge.target_id not in visited:
                    visited.add(edge.target_id)
                    queue.append([*path, edge.target_id])
        return None

    def to_export_dict(self) -> dict:
        return {
            "nodes": [{"id": n.id, "type": n.type.value, "label": n.label, "metadata": n.metadata} for n in self.nodes],
            "edges": [
                {"source": e.source_id, "target": e.target_id, "type": e.edge_type.value, "confidence": e.confidence, "evidence": e.evidence}
                for e in self.edges
            ],
        }


class RootCauseGraphBuilder:
    def __init__(self, analyzer: ErrorAnalyzer | None = None):
        self._analyzer = analyzer or ErrorAnalyzer()

    def build(
        self,
        trace: ExecutionTrace,
        failures: list[DetectedFailure] | None = None,
        patterns: list[FailurePattern] | None = None,
    ) -> RootCauseGraph:
        if failures is None:
            failures = self._analyzer.analyze_trace(trace).failures

        nodes: list[RootCauseNode] = []
        edges: list[RootCauseEdge] = []
        component_node_ids: dict[str, str] = {}

        def _component_node(component: str) -> str:
            if component not in component_node_ids:
                node_id = f"component:{component}"
                component_node_ids[component] = node_id
                nodes.append(RootCauseNode(id=node_id, type=RootCauseNodeType.COMPONENT, label=component))
            return component_node_ids[component]

        execution_node_id = f"execution:{trace.execution_id}"
        nodes.append(RootCauseNode(id=execution_node_id, type=RootCauseNodeType.EXECUTION, label=trace.query, metadata={"confidence": trace.confidence}))

        for component in trace.components():
            _component_node(component)

        failure_node_ids: dict[str, str] = {}
        for i, failure in enumerate(failures):
            failure_node_id = f"failure:{trace.execution_id}:{i}"
            failure_node_ids[failure.category.value] = failure_node_id
            nodes.append(
                RootCauseNode(
                    id=failure_node_id, type=RootCauseNodeType.FAILURE, label=failure.category.value,
                    metadata={"root_cause": failure.root_cause, "severity": failure.severity.value, "confidence": failure.confidence},
                )
            )
            edges.append(RootCauseEdge(execution_node_id, failure_node_id, RootCauseEdgeType.CORRELATED_WITH, 1.0, "This execution produced this failure."))

            if failure.affected_components:
                edges.append(
                    RootCauseEdge(failure_node_id, _component_node(failure.affected_components[0]), RootCauseEdgeType.CAUSED_BY, failure.confidence, failure.root_cause)
                )
            for secondary_component in failure.affected_components[1:]:
                edges.append(
                    RootCauseEdge(failure_node_id, _component_node(secondary_component), RootCauseEdgeType.CONTRIBUTED_TO, round(failure.confidence * 0.5, 4), failure.root_cause)
                )

        for upstream_category in _UPSTREAM_AMPLIFIER_CATEGORIES & failure_node_ids.keys():
            for downstream_category in _DOWNSTREAM_AMPLIFIED_CATEGORIES & failure_node_ids.keys():
                edges.append(
                    RootCauseEdge(
                        failure_node_ids[downstream_category], failure_node_ids[upstream_category], RootCauseEdgeType.AMPLIFIED_BY, 0.5,
                        "Heuristic: upstream retrieval/ranking weakness often coincides with downstream verification/faithfulness problems — a documented correlation, not asserted causality.",
                    )
                )

        if trace.claim_extraction_trace is not None:
            claim_node_id = f"claim:{trace.execution_id}"
            nodes.append(RootCauseNode(claim_node_id, RootCauseNodeType.CLAIM, f"{trace.claim_extraction_trace.get('claim_count', 0)} claim(s)", trace.claim_extraction_trace))
            edges.append(RootCauseEdge(execution_node_id, claim_node_id, RootCauseEdgeType.CORRELATED_WITH, 1.0, "Claim extraction output for this execution."))

        if trace.recommendation_trace is not None:
            recommendation_node_id = f"recommendation:{trace.execution_id}"
            label = f"{trace.recommendation_trace.get('verified_count', 0)} verified / {trace.recommendation_trace.get('rejected_count', 0)} rejected"
            nodes.append(RootCauseNode(recommendation_node_id, RootCauseNodeType.RECOMMENDATION, label, trace.recommendation_trace))
            edges.append(RootCauseEdge(execution_node_id, recommendation_node_id, RootCauseEdgeType.CORRELATED_WITH, 1.0, "Recommendation verification output for this execution."))

        if trace.retrieval_trace is not None:
            evidence_node_id = f"evidence:{trace.execution_id}"
            nodes.append(RootCauseNode(evidence_node_id, RootCauseNodeType.EVIDENCE, f"{trace.retrieval_trace.get('evidence_count', 0)} evidence item(s)", trace.retrieval_trace))
            edges.append(RootCauseEdge(execution_node_id, evidence_node_id, RootCauseEdgeType.CORRELATED_WITH, 1.0, "Retrieval output for this execution."))

        for pattern in patterns or []:
            if pattern.category in failure_node_ids:
                pattern_node_id = f"pattern:{pattern.pattern_id}"
                nodes.append(RootCauseNode(pattern_node_id, RootCauseNodeType.PATTERN, pattern.category, {"support": pattern.support, "frequency": pattern.frequency}))
                edges.append(RootCauseEdge(pattern_node_id, failure_node_ids[pattern.category], RootCauseEdgeType.CORRELATED_WITH, pattern.confidence, pattern.description))

        return RootCauseGraph(nodes=nodes, edges=edges)
