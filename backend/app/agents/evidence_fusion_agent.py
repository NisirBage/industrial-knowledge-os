"""Evidence Fusion agent: merges the Task Graph's retrieval evidence with the
Knowledge Graph agent's expansion evidence into a single EvidenceBundle —
every later agent consumes this bundle, never the raw evidence lists."""

from app.evidence.fusion import EvidenceFusionEngine
from app.orchestrator.state import AgentState


class EvidenceFusionAgent:
    name = "evidence_fusion"

    def __init__(self, fusion_engine: EvidenceFusionEngine):
        self._fusion_engine = fusion_engine

    def run(self, state: AgentState) -> AgentState:
        bundle = self._fusion_engine.fuse(state["task_graph"], state.get("graph_evidence", []))
        # Overwrite the raw pre-fusion evidence list with the deduped, ranked
        # result so ChatResponse.evidence reflects what was actually reasoned over.
        return {**state, "evidence_bundle": bundle, "evidence": bundle.all_evidence()}
