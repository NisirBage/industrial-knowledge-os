"""Recommendation Verification (O2 Component 7).

Structural-only verification already exists: `VerifierAgent`
(app/agents/verifier_agent.py) drops any recommendation whose
`supporting_evidence` chunk ids aren't in the evidence bundle — a
defense-in-depth check, not a quality bar. This module adds the semantic
bar the O2 spec asks for: every recommendation must additionally clear real
evidence/confidence/graph-support thresholds before being presented, using
the SAME Claim Graph Claim Verification already computed upstream — no
second, parallel scoring system.

"minimum prediction confidence" is honestly out of scope for the same
reason as Claim Verification's Vision/Predictions gap (see
app/claims/verification.py): no Prediction Engine call happens inside the
chat pipeline today, so there is no real prediction confidence to threshold
against here — this criterion is a documented no-op until a chat-time
prediction call exists (Roadmap), not a fabricated always-pass silently.

Every recommendation, verified or rejected, gets a `RecommendationVerification`
record: status, the reasons actually checked, and which Claim Graph claim
ids support it — so a caller (frontend, Faithfulness Scoring) can always
answer "why was/wasn't this recommendation shown."
"""

from dataclasses import dataclass, field
from enum import Enum

from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.recommendation.models import Recommendation, RecommendationBundle

_SUPPORTED_VERIFICATION_STATUSES = {ClaimVerificationStatus.SUPPORTED, ClaimVerificationStatus.PARTIALLY_SUPPORTED}
_RISK_LIKE_CLAIM_TYPES = {ClaimType.RISK, ClaimType.HYPOTHESIS}


class RecommendationVerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class RecommendationVerification:
    recommendation_id: str
    status: RecommendationVerificationStatus
    reasons: list[str] = field(default_factory=list)
    supporting_claim_ids: list[str] = field(default_factory=list)
    supporting_risk_claim_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class VerifiedRecommendationBundle:
    verified: list[Recommendation]
    rejected: list[Recommendation]
    verifications: dict[str, RecommendationVerification]  # keyed by recommendation.id, covers verified AND rejected
    overall_confidence: float


@dataclass(frozen=True)
class RecommendationVerificationThresholds:
    min_evidence_count: int = 1
    min_confidence: float = 0.3
    require_graph_support: bool = True


class RecommendationVerifier:
    def __init__(self, thresholds: RecommendationVerificationThresholds | None = None):
        self._thresholds = thresholds or RecommendationVerificationThresholds()

    def verify(self, bundle: RecommendationBundle, claim_graph: ClaimGraph) -> VerifiedRecommendationBundle:
        verified: list[Recommendation] = []
        rejected: list[Recommendation] = []
        verifications: dict[str, RecommendationVerification] = {}

        for recommendation in bundle.recommendations:
            record = self._verify_one(recommendation, claim_graph)
            verifications[recommendation.id] = record
            (verified if record.status == RecommendationVerificationStatus.VERIFIED else rejected).append(recommendation)

        overall_confidence = round(sum(r.confidence for r in verified) / len(verified), 2) if verified else 0.0
        return VerifiedRecommendationBundle(
            verified=verified, rejected=rejected, verifications=verifications, overall_confidence=overall_confidence
        )

    def _verify_one(self, recommendation: Recommendation, claim_graph: ClaimGraph) -> RecommendationVerification:
        reasons: list[str] = []
        supporting_claims = self._supporting_claims(recommendation, claim_graph)
        supporting_risk_claims = [c for c in supporting_claims if c.claim_type in _RISK_LIKE_CLAIM_TYPES]

        evidence_ok = len(recommendation.supporting_evidence) >= self._thresholds.min_evidence_count
        if not evidence_ok:
            reasons.append(
                f"Only {len(recommendation.supporting_evidence)} supporting evidence item(s), "
                f"needs >= {self._thresholds.min_evidence_count}."
            )

        confidence_ok = recommendation.confidence >= self._thresholds.min_confidence
        if not confidence_ok:
            reasons.append(f"Confidence {recommendation.confidence} below minimum {self._thresholds.min_confidence}.")

        graph_ok = True
        if self._thresholds.require_graph_support:
            graph_ok = any(c.verification_status in _SUPPORTED_VERIFICATION_STATUSES for c in supporting_claims)
            if not graph_ok:
                reasons.append("No supporting claim reached SUPPORTED/PARTIALLY_SUPPORTED verification status.")

        verified = evidence_ok and confidence_ok and graph_ok
        status = RecommendationVerificationStatus.VERIFIED if verified else RecommendationVerificationStatus.REJECTED
        if verified:
            reasons = [
                f"Backed by {len(recommendation.supporting_evidence)} evidence item(s), "
                f"confidence {recommendation.confidence}, and {len(supporting_claims)} verified claim(s)."
            ]

        return RecommendationVerification(
            recommendation_id=recommendation.id,
            status=status,
            reasons=reasons,
            supporting_claim_ids=[c.id for c in supporting_claims],
            supporting_risk_claim_ids=[c.id for c in supporting_risk_claims],
            confidence=recommendation.confidence,
        )

    @staticmethod
    def _supporting_claims(recommendation: Recommendation, claim_graph: ClaimGraph) -> list[Claim]:
        evidence_ids = set(recommendation.supporting_evidence)
        return [c for c in claim_graph.claims if evidence_ids & set(c.source_evidence_ids)]
