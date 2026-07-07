"""Planner agent: extracts the query's entities once (reused by every later
agent that needs them) and decomposes the query into an executable Task
Graph. No longer makes an LLM call — planning is deterministic (see
app/planning/planner.py) so it's fast, reproducible, and the LLM is reserved
for the final narration stage only.
"""

from app.documents.chunking import Chunk
from app.entity_extraction.service import EntityExtractionService
from app.orchestrator.state import AgentState
from app.planning.planner import TaskGraphPlanner


class PlannerAgent:
    name = "planner"

    def __init__(self, entity_extraction_service: EntityExtractionService, task_graph_planner: TaskGraphPlanner):
        self._entity_extraction_service = entity_extraction_service
        self._task_graph_planner = task_graph_planner

    def run(self, state: AgentState) -> AgentState:
        query = state["query"]
        query_chunk = Chunk(id="query", document_id="query", text=query, page_number=0, chunk_index=0)
        query_entities = self._entity_extraction_service.extract(query_chunk)
        task_graph = self._task_graph_planner.build(query, query_entities)

        plan_description = "Decomposed into: " + " -> ".join(node.type.value for node in task_graph.topological_order())
        return {**state, "query_entities": query_entities, "task_graph": task_graph, "plan": plan_description}
