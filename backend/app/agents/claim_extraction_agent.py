"""Claim Extraction agent (O2 Components 1/2): converts the fused Evidence
Bundle plus the just-built Industrial Reasoning Graph into the Structured
Claim Graph — runs immediately after `reasoning`, before anything the Claim
Graph is checked against exists downstream."""

from app.claims.extraction import ClaimExtractionEngine
from app.orchestrator.state import AgentState


class ClaimExtractionAgent:
    name = "claim_extraction"

    def __init__(self, claim_extraction_engine: ClaimExtractionEngine):
        self._claim_extraction_engine = claim_extraction_engine

    def run(self, state: AgentState) -> AgentState:
        bundle = state["evidence_bundle"]
        reasoning_graph = state.get("reasoning_graph")
        claim_graph = self._claim_extraction_engine.extract(bundle.all_evidence(), reasoning_graph=reasoning_graph)
        return {**state, "claim_graph": claim_graph}
