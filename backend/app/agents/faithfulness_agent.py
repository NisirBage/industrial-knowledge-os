"""Faithfulness agent (O2 Component 8): the last pipeline stage, scoring the
Narrator's final answer against the verified Claim Graph. Runs after
`report` since it needs the answer text; reads `full_verified_claim_graph`
(every claim, verification status set, nothing dropped) for coverage/
contradiction-rate denominators and `claim_graph` (pruned + propagated) for
what the answer was actually narrated from."""

from app.claims.models import ClaimGraph
from app.faithfulness.scorer import FaithfulnessScorer
from app.orchestrator.state import AgentState


class FaithfulnessAgent:
    name = "faithfulness"

    def __init__(self, faithfulness_scorer: FaithfulnessScorer):
        self._faithfulness_scorer = faithfulness_scorer

    def run(self, state: AgentState) -> AgentState:
        answer = state.get("answer", "")
        full_verified_claim_graph = state.get("full_verified_claim_graph") or ClaimGraph(claims=[], relations=[])
        claim_graph = state.get("claim_graph") or ClaimGraph(claims=[], relations=[])
        bundle = state["evidence_bundle"]
        recommendation_verification = state.get("recommendation_verification")

        report = self._faithfulness_scorer.score(
            answer, full_verified_claim_graph, claim_graph, bundle, recommendation_verification
        )
        return {**state, "faithfulness_report": report}
