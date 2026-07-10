"""Confidence Propagation (O2 Component 5).

Replaces "just average the raw numbers" with an explicit, two-stage,
documented combination — no averaging anywhere in this module:

  1. **Local confidence** — computed per claim from whichever real signals
     are already on it: extraction-time confidence (itself derived from
     Evidence's bm25/vector/graph/reranker/ltr scores for FACT claims, or
     the Reasoning Graph node's own confidence for RISK/HYPOTHESIS claims —
     see app/claims/extraction.py) combined with Claim Verification's own
     signals (verification_status, support_count, contradiction_count) via
     an explicit weighted formula (`_local_confidence`).
  2. **Graph propagation** — a decayed BFS over `ClaimRelation` edges, the
     same shape as Graph Intelligence's own risk-propagation BFS
     (`app/graph_intelligence/propagation.py`: `next_risk = risk * decay`),
     except here propagation *raises* a claim's confidence toward whatever
     its connected, well-supported neighbors show, rather than only
     decaying a single risk value outward from one source.

`PropagationConfig` holds every tunable constant so propagation rules are
configurable, not hardcoded inside the algorithm. `BayesianConfidencePropagator`
is a documented, honest extension point (raises `NotImplementedError`,
same pattern as O1's `NeuralRanker` stub) for the spec's "Support Bayesian
propagation later" — not implemented today.

Vision confidence and OCR confidence are real fields on Vision subsystem
outputs (`app/vision/`) but are not wired into local confidence here: no
Vision call happens inside the chat pipeline today (same honest scope note
as Claim Verification's Vision/Predictions gap) — tracked in the Roadmap.
"""

from dataclasses import dataclass, replace
from typing import Protocol

from app.claims.models import Claim, ClaimGraph, ClaimVerificationStatus

_VERIFICATION_MULTIPLIER: dict[ClaimVerificationStatus, float] = {
    ClaimVerificationStatus.SUPPORTED: 1.15,
    ClaimVerificationStatus.PARTIALLY_SUPPORTED: 1.0,
    ClaimVerificationStatus.CONTRADICTED: 0.2,
    ClaimVerificationStatus.INSUFFICIENT_EVIDENCE: 0.6,
    ClaimVerificationStatus.UNKNOWN: 1.0,
}
_SUPPORT_BONUS_PER_ITEM = 0.03
_SUPPORT_BONUS_CAP = 0.15
_CONTRADICTION_PENALTY_PER_ITEM = 0.25
_NEIGHBOR_INFLUENCE_FACTOR = 0.1


@dataclass(frozen=True)
class PropagationConfig:
    decay_factor: float = 0.5
    max_hops: int = 3
    min_confidence: float = 0.05
    max_confidence: float = 0.99


class ConfidencePropagator(Protocol):
    def propagate(self, claim_graph: ClaimGraph, config: PropagationConfig | None = None) -> ClaimGraph: ...


class GraphDecayConfidencePropagator:
    """Default, real implementation."""

    def propagate(self, claim_graph: ClaimGraph, config: PropagationConfig | None = None) -> ClaimGraph:
        config = config or PropagationConfig()
        local = {claim.id: self._local_confidence(claim) for claim in claim_graph.claims}
        propagated = dict(local)
        adjacency = self._undirected_adjacency(claim_graph)

        for claim in claim_graph.claims:
            self._spread(claim.id, local[claim.id], adjacency, propagated, config)

        updated_claims = [
            replace(
                c,
                propagated_confidence=round(
                    max(config.min_confidence, min(config.max_confidence, propagated[c.id])), 4
                ),
            )
            for c in claim_graph.claims
        ]
        return claim_graph.with_claims(updated_claims)

    @staticmethod
    def _spread(
        origin_id: str,
        origin_confidence: float,
        adjacency: dict[str, set[str]],
        propagated: dict[str, float],
        config: PropagationConfig,
    ) -> None:
        frontier = {origin_id}
        visited = {origin_id}
        influence = origin_confidence

        for _ in range(config.max_hops):
            influence *= config.decay_factor
            if influence < config.min_confidence:
                break
            next_frontier: set[str] = set()
            for node_id in frontier:
                for neighbor_id in adjacency.get(node_id, ()):
                    if neighbor_id in visited or neighbor_id not in propagated:
                        continue
                    visited.add(neighbor_id)
                    next_frontier.add(neighbor_id)
                    propagated[neighbor_id] = min(
                        config.max_confidence, propagated[neighbor_id] + influence * _NEIGHBOR_INFLUENCE_FACTOR
                    )
            frontier = next_frontier
            if not frontier:
                break

    @staticmethod
    def _local_confidence(claim: Claim) -> float:
        multiplier = _VERIFICATION_MULTIPLIER.get(claim.verification_status, 1.0)
        support_bonus = min(_SUPPORT_BONUS_CAP, claim.support_count * _SUPPORT_BONUS_PER_ITEM)
        contradiction_penalty = claim.contradiction_count * _CONTRADICTION_PENALTY_PER_ITEM
        value = claim.confidence * multiplier + support_bonus - contradiction_penalty
        return max(0.0, min(1.0, value))

    @staticmethod
    def _undirected_adjacency(claim_graph: ClaimGraph) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {}
        for relation in claim_graph.relations:
            adjacency.setdefault(relation.source_claim_id, set()).add(relation.target_claim_id)
            adjacency.setdefault(relation.target_claim_id, set()).add(relation.source_claim_id)
        return adjacency


class BayesianConfidencePropagator:
    """Documented extension point (spec: "Support Bayesian propagation
    later") — not implemented, same honest-stub pattern as O1's
    NeuralRanker. Use GraphDecayConfidencePropagator (the default)."""

    def propagate(self, claim_graph: ClaimGraph, config: PropagationConfig | None = None) -> ClaimGraph:
        raise NotImplementedError(
            "Bayesian confidence propagation is a documented O2 extension point, not implemented yet — "
            "use GraphDecayConfidencePropagator (the default)."
        )
