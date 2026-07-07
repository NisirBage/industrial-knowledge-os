import uuid

from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel
from app.planning.models import TaskType
from app.planning.planner import TaskGraphPlanner


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


def test_no_entities_produces_single_general_retrieval_task():
    graph = TaskGraphPlanner().build("What is the plant safety policy?", [])

    assert len(graph.nodes) == 1
    assert graph.nodes[0].type == TaskType.GENERAL_RETRIEVAL
    assert graph.nodes[0].dependencies == []


def test_entity_query_produces_full_rca_task_suite_depending_on_locate():
    graph = TaskGraphPlanner().build("Why has Pump P-101 repeatedly failed?", [_entity("P-101")])

    types = {node.type for node in graph.nodes}
    assert types == {
        TaskType.LOCATE_EQUIPMENT,
        TaskType.RETRIEVE_MAINTENANCE_HISTORY,
        TaskType.RETRIEVE_INCIDENTS,
        TaskType.RETRIEVE_INSPECTIONS,
        TaskType.RETRIEVE_STANDARDS,
        TaskType.RETRIEVE_OEM_MANUAL,
    }

    locate_id = next(n.id for n in graph.nodes if n.type == TaskType.LOCATE_EQUIPMENT)
    for node in graph.nodes:
        if node.type != TaskType.LOCATE_EQUIPMENT:
            assert locate_id in node.dependencies


def test_task_graph_is_deterministic_for_same_input():
    entities = [_entity("P-101")]
    graph_a = TaskGraphPlanner().build("Why has Pump P-101 failed?", entities)
    graph_b = TaskGraphPlanner().build("Why has Pump P-101 failed?", entities)

    assert [n.id for n in graph_a.nodes] == [n.id for n in graph_b.nodes]
    assert [n.type for n in graph_a.nodes] == [n.type for n in graph_b.nodes]
