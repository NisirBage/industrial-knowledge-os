"""Faithfulness Scoring (O2 Component 8).

Measures how much of the Narrator's final answer text is actually backed by
verified Claims — using the same `NLIDetector` Component 6 built (checking
whether each answer sentence is entailed by at least one surviving verified
claim's text), not a second heuristic layered on top. Every number here is
computed from real pipeline state (the Claim Graph before and after
verification, the answer text, the verified recommendation bundle) —
nothing is estimated or asserted without a concrete computation behind it.

Weights in `_overall()` are documented, tunable constants (not fabricated
ground truth) — a standard weighted-composite faithfulness score, the same
kind of explicit-formula approach `RecommendationEngine`/`EvidenceFusionEngine`
already use for their own confidence combinations (see
docs/architecture/retrieval-optimization.md's confidence-combination
inventory) rather than a black-box learned scorer.

Runs one NLI call per (claim, answer-sentence) pair — bounded by demo-scale
claim/answer counts today; a real production deployment with hundreds of
claims would want to pre-filter candidate claims (e.g. by shared entity
ids) before the NLI pass — a documented Roadmap item, not implemented here
since HybridRetriever et al. optimize for hundreds of chunks, not the
handful of claims a single chat answer produces.
"""

import re

from app.claims.models import ClaimGraph, ClaimVerificationStatus
from app.evidence.models import EvidenceBundle
from app.faithfulness.models import FaithfulnessReport, SentenceFaithfulness
from app.nli.base import NLIDetector
from app.nli.models import NLILabel
from app.recommendation.verification import VerifiedRecommendationBundle

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_SENTENCE_WEIGHT = 0.4
_CLAIM_COVERAGE_WEIGHT = 0.25
_GRAPH_WEIGHT = 0.2
_RECOMMENDATION_WEIGHT = 0.15
_CONTRADICTION_PENALTY_MAX = 0.2


class FaithfulnessScorer:
    def __init__(self, nli_detector: NLIDetector):
        self._nli_detector = nli_detector

    def score(
        self,
        answer: str,
        full_verified_claim_graph: ClaimGraph,
        verified_claim_graph: ClaimGraph,
        bundle: EvidenceBundle,
        recommendation_verification: VerifiedRecommendationBundle | None = None,
    ) -> FaithfulnessReport:
        sentence_scores = self._score_sentences(answer, verified_claim_graph)
        hallucinated = sum(1 for s in sentence_scores if not s.entailed)
        sentence_ratio = (len(sentence_scores) - hallucinated) / len(sentence_scores) if sentence_scores else 1.0

        total_claims = len(full_verified_claim_graph.claims)
        kept_claims = len(verified_claim_graph.claims)
        unsupported = total_claims - kept_claims
        claim_coverage = kept_claims / total_claims if total_claims else 1.0

        contradicted = sum(
            1 for c in full_verified_claim_graph.claims if c.verification_status == ClaimVerificationStatus.CONTRADICTED
        )
        contradiction_rate = contradicted / total_claims if total_claims else 0.0

        graph_faithfulness = self._graph_faithfulness(verified_claim_graph)
        recommendation_faithfulness = self._recommendation_faithfulness(recommendation_verification)

        overall = self._overall(sentence_ratio, claim_coverage, graph_faithfulness, recommendation_faithfulness, contradiction_rate)

        return FaithfulnessReport(
            overall_faithfulness=round(overall, 4),
            sentence_faithfulness=sentence_scores,
            recommendation_faithfulness=round(recommendation_faithfulness, 4),
            graph_faithfulness=round(graph_faithfulness, 4),
            claim_coverage=round(claim_coverage, 4),
            unsupported_claim_count=unsupported,
            hallucinated_claim_count=hallucinated,
            missing_evidence_count=len(bundle.missing_information),
            contradiction_rate=round(contradiction_rate, 4),
            summary=self._summary(overall, sentence_scores, hallucinated, kept_claims, total_claims, contradicted),
        )

    def _score_sentences(self, answer: str, claim_graph: ClaimGraph) -> list[SentenceFaithfulness]:
        claim_texts = [(c.id, c.text) for c in claim_graph.claims]
        results: list[SentenceFaithfulness] = []
        for raw_sentence in _SENTENCE_SPLIT.split(answer.strip()):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            best_claim_id: str | None = None
            best_score = 0.0
            entailed = False
            for claim_id, claim_text in claim_texts:
                result = self._nli_detector.classify(claim_text, sentence)
                if result.label == NLILabel.ENTAILMENT and result.score > best_score:
                    best_claim_id, best_score, entailed = claim_id, result.score, True
            results.append(SentenceFaithfulness(sentence=sentence, entailed=entailed, best_claim_id=best_claim_id, score=round(best_score, 4)))
        return results

    @staticmethod
    def _graph_faithfulness(claim_graph: ClaimGraph) -> float:
        if not claim_graph.claims:
            return 1.0
        supported = sum(1 for c in claim_graph.claims if c.verification_status == ClaimVerificationStatus.SUPPORTED)
        return supported / len(claim_graph.claims)

    @staticmethod
    def _recommendation_faithfulness(recommendation_verification: VerifiedRecommendationBundle | None) -> float:
        if recommendation_verification is None:
            return 1.0
        total = len(recommendation_verification.verified) + len(recommendation_verification.rejected)
        if total == 0:
            return 1.0
        return len(recommendation_verification.verified) / total

    @staticmethod
    def _overall(
        sentence_ratio: float,
        claim_coverage: float,
        graph_faithfulness: float,
        recommendation_faithfulness: float,
        contradiction_rate: float,
    ) -> float:
        weighted = (
            _SENTENCE_WEIGHT * sentence_ratio
            + _CLAIM_COVERAGE_WEIGHT * claim_coverage
            + _GRAPH_WEIGHT * graph_faithfulness
            + _RECOMMENDATION_WEIGHT * recommendation_faithfulness
        )
        penalty = contradiction_rate * _CONTRADICTION_PENALTY_MAX
        return max(0.0, min(1.0, weighted - penalty))

    @staticmethod
    def _summary(
        overall: float,
        sentence_scores: list[SentenceFaithfulness],
        hallucinated: int,
        kept_claims: int,
        total_claims: int,
        contradicted: int,
    ) -> str:
        supported_sentences = len(sentence_scores) - hallucinated
        parts = [f"Faithfulness {overall:.2f}: {supported_sentences}/{len(sentence_scores)} sentence(s) entailed by a verified claim,"]
        parts.append(f"{kept_claims}/{total_claims} claim(s) retained after verification,")
        parts.append(f"{contradicted} contradiction(s) found.")
        return " ".join(parts)
