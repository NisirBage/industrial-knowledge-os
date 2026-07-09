"""Task Graph Executor: runs a TaskGraph's nodes in dependency (topological)
order, calling hybrid retrieval and/or the knowledge graph depending on each
node's TaskType, and records status/result/confidence back onto the node.

A node's `status` distinguishes a genuine execution failure (FAILED, caught
exception) from a search that legitimately found nothing (COMPLETED with an
empty result) — the latter is what EvidenceFusionEngine reports as "missing
information," not an error.
"""

import logging
from concurrent.futures import Executor
from typing import Callable

from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import RelationType
from app.planning.models import TaskGraph, TaskNode, TaskStatus, TaskType
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.retrieval.evidence import Evidence
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_TEXT_QUERY_SUFFIX: dict[TaskType, str] = {
    TaskType.RETRIEVE_MAINTENANCE_HISTORY: "maintenance history",
    TaskType.RETRIEVE_INCIDENTS: "incident failure",
    TaskType.RETRIEVE_INSPECTIONS: "inspection records",
    TaskType.RETRIEVE_STANDARDS: "standards regulations compliance",
    TaskType.RETRIEVE_OEM_MANUAL: "OEM manual manufacturer specifications",
}

_GRAPH_HISTORY_CONFIDENCE = 0.5


class TaskGraphExecutor:
    def __init__(self, retriever: HybridRetriever, graph_repository: GraphRepositoryProtocol, executor: Executor | None = None):
        self._retriever = retriever
        self._graph_repository = graph_repository
        # O1 Component 4: when provided, every node within one dependency
        # level (see TaskGraph.levels()) runs concurrently instead of
        # sequentially — `None` (the default) preserves the exact prior
        # one-node-at-a-time behavior.
        self._executor = executor

    def execute(self, task_graph: TaskGraph, query: str, query_entities: list[ExtractedEntity]) -> TaskGraph:
        for level in task_graph.levels():
            if self._executor is None or len(level) == 1:
                for node in level:
                    self._execute_node(node, query, query_entities)
            else:
                futures = {self._executor.submit(self._execute_node, node, query, query_entities): node for node in level}
                for future in futures:
                    future.result()  # propagate any unexpected exception from _execute_node itself
        return task_graph

    def _execute_node(self, node: TaskNode, query: str, query_entities: list[ExtractedEntity]) -> None:
        try:
            node.result = self._run_node(node, query, query_entities)
            node.confidence = max((e.confidence for e in node.result), default=0.0)
            node.status = TaskStatus.COMPLETED
        except Exception:
            logger.exception("Task node %s (%s) failed to execute", node.id, node.type)
            node.result = []
            node.confidence = 0.0
            node.status = TaskStatus.FAILED

    def _run_node(self, node: TaskNode, query: str, query_entities: list[ExtractedEntity]) -> list[Evidence]:
        if node.type == TaskType.GENERAL_RETRIEVAL:
            return self._retriever.retrieve(query, top_k=8)

        if node.type == TaskType.LOCATE_EQUIPMENT:
            return self._locate(query_entities)

        if node.type == TaskType.RETRIEVE_MAINTENANCE_HISTORY:
            return self._graph_history(query_entities, self._graph_repository.maintenance_history) + self._text_retrieve(node.type, query)

        if node.type == TaskType.RETRIEVE_INCIDENTS:
            return self._graph_history(query_entities, self._graph_repository.incident_history) + self._text_retrieve(node.type, query)

        if node.type == TaskType.RETRIEVE_STANDARDS:
            return self._standards_relations(query_entities) + self._text_retrieve(node.type, query)

        if node.type in (TaskType.RETRIEVE_INSPECTIONS, TaskType.RETRIEVE_OEM_MANUAL):
            return self._text_retrieve(node.type, query)

        raise ValueError(f"Unhandled task type: {node.type}")

    def _text_retrieve(self, task_type: TaskType, query: str, top_k: int = 4) -> list[Evidence]:
        suffix = _TEXT_QUERY_SUFFIX.get(task_type, "")
        augmented_query = f"{query} {suffix}".strip()
        return self._retriever.retrieve(augmented_query, top_k=top_k)

    def _locate(self, query_entities: list[ExtractedEntity]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for entity in query_entities:
            node = self._graph_repository.find_entity(entity.canonical_name, entity.label)
            if node is None:
                continue
            evidence.append(
                Evidence(
                    document_id="",
                    page_number=0,
                    chunk_id=f"graph-locate:{node['id']}",
                    text=f"Located {node['type']} '{node['canonical_name']}' in the knowledge graph.",
                    entity_ids=[node["id"]],
                    graph_score=_GRAPH_HISTORY_CONFIDENCE,
                    source="graph",
                    confidence=_GRAPH_HISTORY_CONFIDENCE,
                    agent="knowledge_graph",
                )
            )
        return evidence

    def _graph_history(
        self, query_entities: list[ExtractedEntity], history_method: Callable[[str], list[dict[str, str]]]
    ) -> list[Evidence]:
        evidence: list[Evidence] = []
        for entity in query_entities:
            for record in history_method(entity.canonical_name):
                evidence.append(
                    Evidence(
                        document_id="",
                        page_number=0,
                        chunk_id=f"graph-history:{record['id']}",
                        text=f"{entity.canonical_name} {record['relation_type']} {record['canonical_name']} ({record['type']})",
                        entity_ids=[record["id"]],
                        graph_score=_GRAPH_HISTORY_CONFIDENCE,
                        source="graph",
                        confidence=_GRAPH_HISTORY_CONFIDENCE,
                        agent="knowledge_graph",
                    )
                )
        return evidence

    def _standards_relations(self, query_entities: list[ExtractedEntity]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for entity in query_entities:
            node = self._graph_repository.find_entity(entity.canonical_name, entity.label)
            if node is None:
                continue
            neighbors = self._graph_repository.find_neighbors(node["id"], relation_type=RelationType.USES_STANDARD, depth=1)
            for neighbor in neighbors:
                evidence.append(
                    Evidence(
                        document_id="",
                        page_number=0,
                        chunk_id=f"graph-standard:{node['id']}:{neighbor['id']}",
                        text=f"{entity.canonical_name} USES_STANDARD {neighbor['canonical_name']}",
                        entity_ids=[node["id"], neighbor["id"]],
                        graph_score=_GRAPH_HISTORY_CONFIDENCE,
                        source="graph",
                        confidence=_GRAPH_HISTORY_CONFIDENCE,
                        agent="knowledge_graph",
                    )
                )
        return evidence
