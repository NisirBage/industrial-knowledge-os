"""Claim Verification Engine (O2 Component 3).

Every claim the Claim Extraction Engine produced is checked against real,
already-built sources of truth — never re-derived from thin air:

  - Knowledge Graph: `GraphRepository.find_entity(subject)` for entity
    claims; `find_neighbors()` for whether a claimed relation between two
    entities is actually a persisted graph edge.
  - Retrieved Evidence / Timeline / Maintenance History: how many OTHER
    evidence chunks and `EvidenceBundle.timeline` events (beyond the
    claim's own origin chunk) mention the claim's subject — corroboration
    from independent sources, the same idea O1.2's near-duplicate/entity-
    resolution corroboration already used.
  - Entity Resolution: not a separate lookup here — the Knowledge Graph is
    already deduplicated at ingestion/merge time (`normalize_canonical_name`
    at extraction, `EntityResolver`/`GraphMerger` for explicit merges — see
    O1.2), so an exact `find_entity(subject)` match already reflects any
    prior entity-resolution outcome by construction.

Vision Results and Predictions are honestly out of scope for this class:
neither the Vision Knowledge Graph nor the Prediction Engine is invoked
inside the chat pipeline today (see docs/architecture/reasoning.md) —
wiring those in without an actual chat-time call to either would be
fabricated corroboration, not real verification. Tracked in the Roadmap.

Status thresholds are named, not fabricated:
  CONTRADICTED           – the claim's origin evidence overlaps a
                            Contradiction the (possibly NLI-augmented, see
                            O2 Component 6) ContradictionDetector already
                            found for this bundle.
  SUPPORTED               – Knowledge Graph support AND at least one
                            corroborating source beyond the claim's origin.
  PARTIALLY_SUPPORTED     – graph support OR corroboration, not both.
  INSUFFICIENT_EVIDENCE   – neither graph support nor corroboration exists.
  UNKNOWN                 – not a verifiable claim type here (RECOMMENDATION
                            claims are verified by Recommendation
                            Verification, app/recommendation/verification.py;
                            PREDICTION claims have no chat-time source yet).

Per the spec's "unsupported claims never reach the narrator":
`drop_unsupported()` removes CONTRADICTED / INSUFFICIENT_EVIDENCE / UNKNOWN
claims (and any ClaimRelation touching them) from the graph handed onward;
SUPPORTED and PARTIALLY_SUPPORTED survive, still visibly labeled.
"""

from dataclasses import replace

from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.evidence.models import EvidenceBundle
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import GraphRepositoryProtocol, entity_id_for

_VERIFIABLE_TYPES = {ClaimType.FACT, ClaimType.RISK, ClaimType.HYPOTHESIS, ClaimType.OBSERVATION}
_UNSUPPORTED_STATUSES = {
    ClaimVerificationStatus.CONTRADICTED,
    ClaimVerificationStatus.INSUFFICIENT_EVIDENCE,
    ClaimVerificationStatus.UNKNOWN,
}
# Structural ClaimRelation labels added by extraction that are not real
# ontology relation types — never worth a Knowledge Graph confirmation check.
_NON_ONTOLOGY_RELATION_LABELS = {"SUPPORTS", "ABOUT"}


class ClaimVerificationEngine:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def verify(self, claim_graph: ClaimGraph, bundle: EvidenceBundle) -> ClaimGraph:
        relation_confirmations = self._confirmed_relation_claim_ids(claim_graph)
        verified_claims = [
            self._verify_claim(claim, bundle, relation_confirmations.get(claim.id, 0))
            for claim in claim_graph.claims
        ]
        return claim_graph.with_claims(verified_claims)

    def drop_unsupported(self, claim_graph: ClaimGraph) -> ClaimGraph:
        kept = [c for c in claim_graph.claims if c.verification_status not in _UNSUPPORTED_STATUSES]
        kept_ids = {c.id for c in kept}
        kept_relations = [
            r for r in claim_graph.relations if r.source_claim_id in kept_ids and r.target_claim_id in kept_ids
        ]
        return ClaimGraph(claims=kept, relations=kept_relations)

    def _verify_claim(self, claim: Claim, bundle: EvidenceBundle, relation_confirmations: int) -> Claim:
        if claim.claim_type not in _VERIFIABLE_TYPES:
            return claim

        graph_support = self._graph_supports(claim) or relation_confirmations > 0
        corroboration_count = self._corroboration_count(claim, bundle) + relation_confirmations
        contradicted = self._is_contradicted(claim, bundle)

        if contradicted:
            status = ClaimVerificationStatus.CONTRADICTED
            notes = "Conflicts with another claim's evidence in this bundle (see Contradiction Detection)."
        elif graph_support and corroboration_count > 0:
            status = ClaimVerificationStatus.SUPPORTED
            notes = f"Confirmed in the Knowledge Graph and corroborated by {corroboration_count} additional source(s)."
        elif graph_support or corroboration_count > 0:
            status = ClaimVerificationStatus.PARTIALLY_SUPPORTED
            notes = (
                "Confirmed in the Knowledge Graph, no independent corroboration yet."
                if graph_support
                else f"Corroborated by {corroboration_count} additional source(s), not yet in the Knowledge Graph."
            )
        else:
            status = ClaimVerificationStatus.INSUFFICIENT_EVIDENCE
            notes = "No corroborating evidence or Knowledge Graph entry found beyond the claim's own source."

        return replace(
            claim,
            verification_status=status,
            verification_notes=notes,
            support_count=corroboration_count,
            contradiction_count=1 if contradicted else 0,
        )

    def _graph_supports(self, claim: Claim) -> bool:
        try:
            return self._graph_repository.find_entity(claim.subject) is not None
        except Exception:
            return False

    @staticmethod
    def _corroboration_count(claim: Claim, bundle: EvidenceBundle) -> int:
        origin_ids = set(claim.source_evidence_ids)
        subject_lower = claim.subject.lower()
        count = 0
        for item in bundle.all_evidence():
            if item.chunk_id in origin_ids:
                continue
            if subject_lower in item.text.lower():
                count += 1
        for event in bundle.timeline:
            if event.source_chunk_id in origin_ids:
                continue
            if subject_lower in event.description.lower():
                count += 1
        return count

    @staticmethod
    def _is_contradicted(claim: Claim, bundle: EvidenceBundle) -> bool:
        origin_ids = set(claim.source_evidence_ids)
        return any(origin_ids & set(c.conflicting_evidence) for c in bundle.contradictions)

    def _confirmed_relation_claim_ids(self, claim_graph: ClaimGraph) -> dict[str, int]:
        """For each ClaimRelation whose label is a real ontology RelationType
        (not a structural SUPPORTS/ABOUT link), check whether the Knowledge
        Graph actually has that edge between the two claims' subjects — and
        if so, count it as a Knowledge-Graph confirmation for BOTH endpoint
        claims (source and target), since a confirmed relationship
        corroborates both entities' facts, not just one."""
        confirmations: dict[str, int] = {}
        for relation in claim_graph.relations:
            if relation.relation_type in _NON_ONTOLOGY_RELATION_LABELS:
                continue
            source_claim = claim_graph.claim_by_id(relation.source_claim_id)
            target_claim = claim_graph.claim_by_id(relation.target_claim_id)
            if source_claim is None or target_claim is None:
                continue
            if not self._graph_confirms_relation(source_claim, target_claim, relation.relation_type):
                continue
            confirmations[source_claim.id] = confirmations.get(source_claim.id, 0) + 1
            confirmations[target_claim.id] = confirmations.get(target_claim.id, 0) + 1
        return confirmations

    def _graph_confirms_relation(self, source_claim: Claim, target_claim: Claim, relation_type_value: str) -> bool:
        try:
            source_label = EntityLabel(source_claim.object)
            relation_type = RelationType(relation_type_value)
        except ValueError:
            return False

        source_id = entity_id_for(source_label, source_claim.subject)
        try:
            neighbors = self._graph_repository.find_neighbors(source_id, relation_type=relation_type, depth=1)
        except Exception:
            return False
        return any(n.get("canonical_name") == target_claim.subject for n in neighbors)
