"""Recommendation Verification agent (O2 Component 7): runs after the
existing structural `verifier` stage, applying the semantic evidence/
confidence/graph-support thresholds from app/recommendation/verification.py.
Rejected recommendations are dropped from `recommendation_bundle` (so every
downstream reader — the Narrator, ReportAgent's recommended_actions —
only ever sees verified ones); the full verified+rejected record survives
separately in `recommendation_verification` for the API/frontend/
Faithfulness Scoring to inspect."""

from app.orchestrator.state import AgentState
from app.recommendation.models import RecommendationBundle
from app.recommendation.verification import RecommendationVerifier


class RecommendationVerificationAgent:
    name = "recommendation_verification"

    def __init__(self, recommendation_verifier: RecommendationVerifier):
        self._recommendation_verifier = recommendation_verifier

    def run(self, state: AgentState) -> AgentState:
        recommendation_bundle = state.get("recommendation_bundle") or RecommendationBundle(recommendations=[], overall_confidence=0.5)
        claim_graph = state["claim_graph"]

        result = self._recommendation_verifier.verify(recommendation_bundle, claim_graph)
        updated_bundle = RecommendationBundle(recommendations=result.verified, overall_confidence=result.overall_confidence)

        return {**state, "recommendation_bundle": updated_bundle, "recommendation_verification": result}
