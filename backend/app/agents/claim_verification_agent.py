"""Claim Verification agent (O2 Component 3): checks every extracted claim
against the Knowledge Graph and corroborating evidence, then drops
unsupported claims — "unsupported claims never reach the narrator."

Keeps BOTH the full verified graph (`full_verified_claim_graph`, every
claim with its verification_status set, nothing dropped — what Faithfulness
Scoring needs to compute contradiction_rate/claim_coverage against the true
total) and the pruned graph (`claim_graph`, only SUPPORTED/
PARTIALLY_SUPPORTED claims — what Confidence Propagation, Recommendation
Verification, Evidence Attribution, and the Narrator all consume).
"""

from app.claims.verification import ClaimVerificationEngine
from app.orchestrator.state import AgentState


class ClaimVerificationAgent:
    name = "claim_verification"

    def __init__(self, claim_verification_engine: ClaimVerificationEngine):
        self._claim_verification_engine = claim_verification_engine

    def run(self, state: AgentState) -> AgentState:
        claim_graph = state["claim_graph"]
        bundle = state["evidence_bundle"]

        full_verified = self._claim_verification_engine.verify(claim_graph, bundle)
        kept = self._claim_verification_engine.drop_unsupported(full_verified)

        return {**state, "claim_graph": kept, "full_verified_claim_graph": full_verified}
