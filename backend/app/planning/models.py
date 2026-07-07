"""Task Graph: a DAG the Planner produces to decompose a query into explicit,
independently-executable retrieval subtasks (per M3's Task Graph Planner
component). Every node ultimately produces `list[Evidence]` — deliberately
scoped to the retrieval-decomposition part of the target pipeline; timeline
building, reasoning, and recommendations are separate downstream stages
(app/evidence, app/reasoning, app/recommendation) that consume the fused
result of this graph rather than living as heterogeneous nodes inside it.

TaskNode is a mutable dataclass (not frozen, unlike Evidence) because it is
an execution record the TaskGraphExecutor updates in place as it works
through the DAG — status/result/confidence change over the node's lifetime,
whereas Evidence is a fusion-ready value object.
"""

from dataclasses import dataclass, field
from enum import Enum

from app.retrieval.evidence import Evidence


class TaskType(str, Enum):
    LOCATE_EQUIPMENT = "locate_equipment"
    RETRIEVE_MAINTENANCE_HISTORY = "retrieve_maintenance_history"
    RETRIEVE_INCIDENTS = "retrieve_incidents"
    RETRIEVE_INSPECTIONS = "retrieve_inspections"
    RETRIEVE_STANDARDS = "retrieve_standards"
    RETRIEVE_OEM_MANUAL = "retrieve_oem_manual"
    GENERAL_RETRIEVAL = "general_retrieval"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    id: str
    type: TaskType
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class TaskGraph:
    nodes: list[TaskNode]

    def get(self, node_id: str) -> TaskNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(f"No task node with id {node_id!r}")

    def topological_order(self) -> list[TaskNode]:
        """Kahn's algorithm. Raises ValueError on a dependency cycle —
        planner-generated graphs should never produce one, so a cycle here
        indicates a planner bug, not bad input data."""
        remaining = {node.id: set(node.dependencies) for node in self.nodes}
        ordered: list[TaskNode] = []

        while remaining:
            ready_ids = [node_id for node_id, deps in remaining.items() if not deps]
            if not ready_ids:
                raise ValueError("Task graph has a dependency cycle")
            for node_id in ready_ids:
                ordered.append(self.get(node_id))
                del remaining[node_id]
            for deps in remaining.values():
                deps.difference_update(ready_ids)

        return ordered

    def all_completed(self) -> bool:
        return all(node.status == TaskStatus.COMPLETED for node in self.nodes)

    def flattened_evidence(self) -> list[Evidence]:
        return [evidence for node in self.nodes for evidence in node.result]
