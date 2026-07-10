"""Structured Claim Graph (O2 Component 1).

Distinct from the Industrial Reasoning Graph (M3): the Reasoning Graph is
evidence-structural — fact/cause/effect/risk nodes derived mechanically from
chunk text and graph-relation triples, one FACT node per evidence item. The
Claim Graph is what O2 adds on top of that: atomic, individually-verifiable
factual statements ("Equipment = Pump P-101", "Failure = Cavitation")
decomposed FROM evidence text via entity/relation extraction, each one
carrying its own confidence, source offsets, and (once Claim Verification
and Confidence Propagation run) a verification status and propagated
confidence. The Narrator (O2) generates from verified Claims only — never
directly from raw evidence text — so an unsupported or contradicted
statement never reaches the final answer.

Frozen dataclasses throughout, same convention as Evidence/Contradiction/
ReasoningNode: every pipeline stage that updates a Claim (verification,
propagation) returns a *new* Claim via `dataclasses.replace`, never mutates
one in place. `ClaimGraph` itself stays mutable only during construction
(add_claim/add_relation, mirroring ReasoningGraph) — once built, later
stages call `with_claims()` to produce a new graph rather than editing
`.claims` directly, so a claim graph handed to one stage is never silently
changed out from under a caller still holding a reference to it.
"""

from dataclasses import dataclass, field
from enum import Enum


class ClaimType(str, Enum):
    FACT = "fact"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    RECOMMENDATION = "recommendation"
    RISK = "risk"
    PREDICTION = "prediction"
    UNKNOWN = "unknown"


class ClaimVerificationStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ClaimSpan:
    """Exact source offset a claim was extracted from — the "exact source
    offsets" the O2 spec requires, absent from Evidence itself today."""

    chunk_id: str
    document_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Claim:
    id: str
    claim_type: ClaimType
    text: str  # human-readable atomic statement, e.g. "Equipment = Pump P-101"
    subject: str
    predicate: str
    object: str = ""
    confidence: float = 0.0  # extraction-time confidence, pre-verification/propagation
    source_evidence_ids: list[str] = field(default_factory=list)  # chunk ids
    source_spans: list[ClaimSpan] = field(default_factory=list)
    timestamp: str = ""
    support_count: int = 0
    contradiction_count: int = 0
    verification_status: ClaimVerificationStatus = ClaimVerificationStatus.UNKNOWN
    verification_notes: str = ""
    # None until Confidence Propagation (Component 5) runs; distinguishes
    # "not yet propagated" from "propagated to exactly the same value."
    propagated_confidence: float | None = None


@dataclass(frozen=True)
class ClaimRelation:
    id: str
    source_claim_id: str
    target_claim_id: str
    # String, not RelationType, so a claim relation can also carry NLI labels
    # (ENTAILS/CONTRADICTS) alongside reused ontology relation types
    # (CAUSES, RELATED_TO, ...) without a second enum.
    relation_type: str
    confidence: float = 0.0


@dataclass
class ClaimGraph:
    claims: list[Claim]
    relations: list[ClaimRelation]

    def add_claim(self, claim: Claim) -> None:
        self.claims.append(claim)

    def add_relation(self, relation: ClaimRelation) -> None:
        self.relations.append(relation)

    def claims_of_type(self, claim_type: ClaimType) -> list[Claim]:
        return [c for c in self.claims if c.claim_type == claim_type]

    def claim_by_id(self, claim_id: str) -> Claim | None:
        for claim in self.claims:
            if claim.id == claim_id:
                return claim
        return None

    def relations_from(self, claim_id: str) -> list[ClaimRelation]:
        return [r for r in self.relations if r.source_claim_id == claim_id]

    def relations_to(self, claim_id: str) -> list[ClaimRelation]:
        return [r for r in self.relations if r.target_claim_id == claim_id]

    def with_claims(self, claims: list[Claim]) -> "ClaimGraph":
        """Return a new ClaimGraph with these claims and the same relations —
        the update path every post-extraction stage (verification,
        propagation) uses instead of mutating `.claims` in place."""
        return ClaimGraph(claims=claims, relations=list(self.relations))
