"""Task Graph Planner: decomposes a query into explicit, dependency-ordered
retrieval subtasks instead of routing it to a single retrieval call.

Deterministic and rule-based (not LLM-driven) on purpose: planning must be
reproducible for testing and must not add LLM latency/cost to every query.
The extension point for LLM-assisted decomposition (e.g. dynamically
inventing new task types) is this class's `build` method — swap its body
without touching TaskGraphExecutor or anything downstream.

When the query names a known entity, the full RCA-style subtask suite runs
(maintenance history, incidents, inspections, standards, OEM manual) since
an equipment-specific question benefits from comprehensive evidence
gathering, matching the worked example in the M3 spec. A query with no
resolvable entity falls back to a single general-retrieval task.
"""

from app.entity_extraction.schemas import ExtractedEntity
from app.planning.models import TaskGraph, TaskNode, TaskType

_LOCATE_ID = "locate_equipment"


class TaskGraphPlanner:
    def build(self, query: str, query_entities: list[ExtractedEntity]) -> TaskGraph:
        if not query_entities:
            return TaskGraph(
                nodes=[
                    TaskNode(
                        id="general_retrieval",
                        type=TaskType.GENERAL_RETRIEVAL,
                        description=f"General document retrieval for: {query}",
                        dependencies=[],
                    )
                ]
            )

        entity_names = ", ".join(e.canonical_name for e in query_entities)
        nodes = [
            TaskNode(
                id=_LOCATE_ID,
                type=TaskType.LOCATE_EQUIPMENT,
                description=f"Locate entities in the knowledge graph: {entity_names}",
                dependencies=[],
            ),
            TaskNode(
                id="retrieve_maintenance_history",
                type=TaskType.RETRIEVE_MAINTENANCE_HISTORY,
                description=f"Retrieve maintenance history for {entity_names}",
                dependencies=[_LOCATE_ID],
            ),
            TaskNode(
                id="retrieve_incidents",
                type=TaskType.RETRIEVE_INCIDENTS,
                description=f"Retrieve incident reports for {entity_names}",
                dependencies=[_LOCATE_ID],
            ),
            TaskNode(
                id="retrieve_inspections",
                type=TaskType.RETRIEVE_INSPECTIONS,
                description=f"Retrieve inspection records for {entity_names}",
                dependencies=[_LOCATE_ID],
            ),
            TaskNode(
                id="retrieve_standards",
                type=TaskType.RETRIEVE_STANDARDS,
                description=f"Retrieve applicable standards and regulations for {entity_names}",
                dependencies=[_LOCATE_ID],
            ),
            TaskNode(
                id="retrieve_oem_manual",
                type=TaskType.RETRIEVE_OEM_MANUAL,
                description=f"Retrieve OEM manual guidance for {entity_names}",
                dependencies=[_LOCATE_ID],
            ),
        ]
        return TaskGraph(nodes=nodes)
