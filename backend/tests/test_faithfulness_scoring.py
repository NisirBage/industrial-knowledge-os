from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.evidence.models import EvidenceBundle
from app.faithfulness.scorer import FaithfulnessScorer
from app.nli.models import NLILabel, NLIResult
from app.recommendation.verification import RecommendationVerification, RecommendationVerificationStatus, VerifiedRecommendationBundle


class _StubNLIDetector:
    """Deterministic NLI stub: entails only exact-text matches (a claim's
    text entails an answer sentence with identical text) — isolates the
    scorer's aggregation logic from any NLI model's own heuristics."""

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        if premise.strip() == hypothesis.strip():
            return NLIResult(premise, hypothesis, NLILabel.ENTAILMENT, 0.95)
        return NLIResult(premise, hypothesis, NLILabel.NEUTRAL, 0.5)


def _claim(claim_id: str, text: str, status: ClaimVerificationStatus = ClaimVerificationStatus.SUPPORTED) -> Claim:
    return Claim(id=claim_id, claim_type=ClaimType.FACT, text=text, subject=text, predicate="is_a", verification_status=status)


def _bundle(missing_information: list[str] | None = None) -> EvidenceBundle:
    return EvidenceBundle(
        evidence=[], graph_evidence=[], timeline=[], metadata=[],
        missing_information=missing_information or [], confidence=0.7,
    )


def test_all_sentences_entailed_gives_full_sentence_credit():
    claim_graph = ClaimGraph(claims=[_claim("c1", "Pump P-101 is running.")], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("Pump P-101 is running.", claim_graph, claim_graph, _bundle())

    assert report.hallucinated_claim_count == 0
    assert report.sentence_faithfulness[0].entailed is True
    assert report.sentence_faithfulness[0].best_claim_id == "c1"


def test_unentailed_sentence_counts_as_hallucinated():
    claim_graph = ClaimGraph(claims=[_claim("c1", "Pump P-101 is running.")], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("Something totally unrelated happened.", claim_graph, claim_graph, _bundle())

    assert report.hallucinated_claim_count == 1
    assert report.sentence_faithfulness[0].entailed is False


def test_empty_answer_produces_no_sentences_and_no_hallucination():
    claim_graph = ClaimGraph(claims=[], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("", claim_graph, claim_graph, _bundle())

    assert report.sentence_faithfulness == []
    assert report.hallucinated_claim_count == 0


def test_claim_coverage_reflects_claims_dropped_by_verification():
    pre = ClaimGraph(claims=[_claim("c1", "a"), _claim("c2", "b"), _claim("c3", "c")], relations=[])
    post = ClaimGraph(claims=[_claim("c1", "a")], relations=[])  # 2 of 3 dropped
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("a", pre, post, _bundle())

    assert report.claim_coverage == round(1 / 3, 4)
    assert report.unsupported_claim_count == 2


def test_contradiction_rate_counts_contradicted_claims_in_pre_verification_graph():
    pre = ClaimGraph(
        claims=[
            _claim("c1", "a", status=ClaimVerificationStatus.SUPPORTED),
            _claim("c2", "b", status=ClaimVerificationStatus.CONTRADICTED),
        ],
        relations=[],
    )
    post = ClaimGraph(claims=[_claim("c1", "a", status=ClaimVerificationStatus.SUPPORTED)], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("a", pre, post, _bundle())

    assert report.contradiction_rate == 0.5


def test_graph_faithfulness_is_fraction_of_kept_claims_that_are_fully_supported():
    post = ClaimGraph(
        claims=[
            _claim("c1", "a", status=ClaimVerificationStatus.SUPPORTED),
            _claim("c2", "b", status=ClaimVerificationStatus.PARTIALLY_SUPPORTED),
        ],
        relations=[],
    )
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("a", post, post, _bundle())

    assert report.graph_faithfulness == 0.5


def test_missing_evidence_count_comes_from_the_bundle():
    claim_graph = ClaimGraph(claims=[], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("", claim_graph, claim_graph, _bundle(missing_information=["gap 1", "gap 2"]))

    assert report.missing_evidence_count == 2


def test_recommendation_faithfulness_defaults_to_one_when_not_provided():
    claim_graph = ClaimGraph(claims=[], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("", claim_graph, claim_graph, _bundle(), recommendation_verification=None)

    assert report.recommendation_faithfulness == 1.0


def test_recommendation_faithfulness_reflects_verified_vs_rejected_ratio():
    verification = VerifiedRecommendationBundle(
        verified=["r1"],
        rejected=["r2", "r3"],
        verifications={
            "r1": RecommendationVerification("r1", RecommendationVerificationStatus.VERIFIED),
            "r2": RecommendationVerification("r2", RecommendationVerificationStatus.REJECTED),
            "r3": RecommendationVerification("r3", RecommendationVerificationStatus.REJECTED),
        },
        overall_confidence=0.5,
    )
    claim_graph = ClaimGraph(claims=[], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("", claim_graph, claim_graph, _bundle(), recommendation_verification=verification)

    assert report.recommendation_faithfulness == round(1 / 3, 4)


def test_summary_string_reports_real_counts():
    claim_graph = ClaimGraph(claims=[_claim("c1", "Pump P-101 is running.")], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("Pump P-101 is running.", claim_graph, claim_graph, _bundle())

    assert "1/1 sentence(s)" in report.summary
    assert "1/1 claim(s) retained" in report.summary
    assert "0 contradiction(s)" in report.summary


def test_multiple_sentences_mixed_entailment_reports_partial_hallucination():
    claim_graph = ClaimGraph(claims=[_claim("c1", "Pump P-101 is running.")], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("Pump P-101 is running. Something totally unrelated happened.", claim_graph, claim_graph, _bundle())

    assert len(report.sentence_faithfulness) == 2
    assert report.hallucinated_claim_count == 1
    assert report.sentence_faithfulness[0].entailed is True
    assert report.sentence_faithfulness[1].entailed is False


def test_best_claim_id_picks_the_highest_scoring_entailing_claim():
    class _ScoredNLIDetector:
        def classify(self, premise, hypothesis):
            score = 0.6 if premise == "weak match" else 0.95
            return NLIResult(premise, hypothesis, NLILabel.ENTAILMENT, score)

    claim_graph = ClaimGraph(
        claims=[
            _claim("weak", "weak match"),
            _claim("strong", "strong match"),
        ],
        relations=[],
    )
    scorer = FaithfulnessScorer(_ScoredNLIDetector())

    report = scorer.score("the answer sentence", claim_graph, claim_graph, _bundle())

    assert report.sentence_faithfulness[0].best_claim_id == "strong"


def test_overall_faithfulness_is_bounded_between_zero_and_one():
    claim_graph = ClaimGraph(claims=[_claim("c1", "a", status=ClaimVerificationStatus.CONTRADICTED)], relations=[])
    scorer = FaithfulnessScorer(_StubNLIDetector())

    report = scorer.score("completely unrelated text", claim_graph, ClaimGraph(claims=[], relations=[]), _bundle())

    assert 0.0 <= report.overall_faithfulness <= 1.0
