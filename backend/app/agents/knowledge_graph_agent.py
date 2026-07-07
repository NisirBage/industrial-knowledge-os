"""Knowledge Graph agent: broader graph-only expansion beyond what
individual task-graph subtasks looked up (general 1-hop neighbors of every
query entity, e.g. CONNECTED_TO relations a maintenance-history-specific
lookup wouldn't surface). Never touches embeddings — per the M3 spec,
"Knowledge Graph Agent... Never uses embeddings unless necessary."
"""

from app.orchestrator.state import AgentState
from app.retrieval.graph_expansion import GraphExpansionStage


class KnowledgeGraphAgent:
    name = "knowledge_graph"

    def __init__(self, graph_expansion: GraphExpansionStage):
        self._graph_expansion = graph_expansion

    def run(self, state: AgentState) -> AgentState:
        expansion = self._graph_expansion.expand_from_entities(state.get("query_entities", []))
        return {**state, "graph_evidence": expansion.graph_evidence}
