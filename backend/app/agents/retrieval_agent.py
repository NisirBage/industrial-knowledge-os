"""Retrieval agent: executes the Planner's Task Graph (hybrid search + direct
graph lookups per subtask) rather than a single flat retrieve() call."""

from app.orchestrator.state import AgentState
from app.planning.executor import TaskGraphExecutor


class RetrievalAgent:
    name = "retrieval"

    def __init__(self, executor: TaskGraphExecutor):
        self._executor = executor

    def run(self, state: AgentState) -> AgentState:
        task_graph = self._executor.execute(state["task_graph"], state["query"], state.get("query_entities", []))
        return {**state, "task_graph": task_graph, "evidence": task_graph.flattened_evidence()}
