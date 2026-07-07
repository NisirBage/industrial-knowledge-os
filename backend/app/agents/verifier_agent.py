"""Verifier agent: the last check before narration. Confirms every
recommendation cites evidence that genuinely exists in the fused bundle
(defense in depth — RecommendationEngine should never produce an
unsupported recommendation, but this is what actually enforces it) and
records what it checked so the check itself is auditable.
"""

import dataclasses

from app.orchestrator.state import AgentState
from app.recommendation.models import RecommendationBundle


class VerifierAgent:
    name = "verifier"

    def run(self, state: AgentState) -> AgentState:
        bundle = state["evidence_bundle"]
        known_chunk_ids = {item.chunk_id for item in bundle.all_evidence()}
        recommendation_bundle = state.get("recommendation_bundle") or RecommendationBundle(recommendations=[], overall_confidence=0.5)

        notes: list[str] = []
        verified_recommendations = []
        for recommendation in recommendation_bundle.recommendations:
            if not recommendation.supporting_evidence:
                notes.append(f"Dropped recommendation '{recommendation.id}': no supporting evidence.")
                continue
            unknown_refs = [c for c in recommendation.supporting_evidence if c not in known_chunk_ids]
            if unknown_refs:
                notes.append(f"Dropped recommendation '{recommendation.id}': references evidence not in the bundle ({unknown_refs}).")
                continue
            verified_recommendations.append(recommendation)

        if state.get("contradictions") and not any("contradiction" in r.id for r in verified_recommendations):
            notes.append("Contradictions were detected but none produced a clarification recommendation.")

        if not notes:
            notes.append(f"Verified {len(verified_recommendations)} recommendation(s) against {len(known_chunk_ids)} evidence item(s); all traceable.")

        verified_bundle = dataclasses.replace(recommendation_bundle, recommendations=verified_recommendations)
        return {**state, "recommendation_bundle": verified_bundle, "verification_notes": notes}
