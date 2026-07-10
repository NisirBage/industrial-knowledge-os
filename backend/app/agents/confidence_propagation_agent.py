"""Confidence Propagation agent (O2 Component 5): spreads verified claims'
confidence across the (already-pruned) Claim Graph via a decayed BFS over
ClaimRelation edges — see app/claims/propagation.py for the algorithm."""

from app.claims.propagation import ConfidencePropagator
from app.orchestrator.state import AgentState


class ConfidencePropagationAgent:
    name = "confidence_propagation"

    def __init__(self, confidence_propagator: ConfidencePropagator):
        self._confidence_propagator = confidence_propagator

    def run(self, state: AgentState) -> AgentState:
        claim_graph = state["claim_graph"]
        propagated = self._confidence_propagator.propagate(claim_graph)
        return {**state, "claim_graph": propagated}
