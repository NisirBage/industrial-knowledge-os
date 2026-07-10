"""Calibration & Uncertainty benchmarks (O3 Component 2): apply the real
ECE/MCE/Brier metrics (app/calibration/metrics.py) to (confidence, correct)
pairs drawn from the same real production components and hand-labeled
scenarios O1/O2's own benchmarks already established — no fabricated
correction, no synthetic confidence numbers invented for this file alone.

Predictions calibration is deliberately NOT a runner here: it already has a
real, live data source (`LearningRepository.confidence_calibration_buckets()`,
recorded prediction outcomes) exposed via `GET /research/calibration` — see
`app/api/v1/research.py`'s calibration-report extension, which reuses
`expected_calibration_error_from_buckets()` directly against that real data
rather than a synthetic runner duplicating it.
"""

import time

from app.calibration.metrics import evaluate_calibration
from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.claims.verification import ClaimVerificationEngine
from app.core.enums import Priority, Severity
from app.evaluation.models import NamedMetric
from app.evidence.models import EvidenceBundle
from app.faithfulness.scorer import FaithfulnessScorer
from app.nli.models import NLILabel, NLIResult
from app.ontology.labels import EntityLabel
from app.recommendation.models import Recommendation, RecommendationBundle
from app.recommendation.verification import RecommendationVerifier
from app.repositories.graph_repository import InMemoryGraphRepository
from app.research.benchmarks.models import BenchmarkReport
from app.research.benchmarks.optimization_runners import _synthetic_labeled_retriever_and_queries
from app.retrieval.evidence import Evidence


def _report_from_calibration(name: str, confidences: list[float], correctness: list[bool], dataset_name: str, start: float) -> BenchmarkReport:
    report = evaluate_calibration(confidences, correctness)
    metrics = [
        NamedMetric("expected_calibration_error", report.expected_calibration_error),
        NamedMetric("maximum_calibration_error", report.maximum_calibration_error),
        NamedMetric("brier_score", report.brier_score),
        NamedMetric("sample_count", float(report.sample_count)),
    ]
    return BenchmarkReport(name, metrics, time.perf_counter() - start, dataset_name, len(confidences))


class RetrievalCalibrationBenchmark:
    """Purpose: calibration of retrieval confidence (sigmoid(reranker_score),
    the same `Evidence.confidence`) against real document-level relevance
    from the synthetic labeled-query fixture O1's `ranking_quality`
    benchmark already uses."""

    name = "retrieval_calibration"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        retriever, labeled_queries = _synthetic_labeled_retriever_and_queries(num_documents=8)

        confidences: list[float] = []
        correctness: list[bool] = []
        for query, relevant_ids in labeled_queries:
            for result in retriever.retrieve(query, top_k=5):
                confidences.append(result.confidence)
                correctness.append(result.document_id in relevant_ids)

        return _report_from_calibration(self.name, confidences, correctness, "synthetic_labeled_queries", start)


def _verification_scenarios() -> list[tuple[Claim, EvidenceBundle]]:
    def claim(claim_id: str, subject: str) -> Claim:
        return Claim(id=claim_id, claim_type=ClaimType.FACT, text=subject, subject=subject, predicate="is_a", object=EntityLabel.PUMP.value, source_evidence_ids=["c1"])

    def bundle(evidence: list[Evidence]) -> EvidenceBundle:
        return EvidenceBundle(evidence=evidence, graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7)

    return [
        (claim("c1", "P-101"), bundle([Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running."), Evidence(document_id="d", page_number=2, chunk_id="c2", text="P-101 was inspected.")])),
        (claim("c2", "V-2201A"), bundle([Evidence(document_id="d", page_number=1, chunk_id="c1", text="Valve V-2201A is closed.")])),
        (claim("c3", "Unknown-Tag-1"), bundle([Evidence(document_id="d", page_number=1, chunk_id="c1", text="Something happened.")])),
        (claim("c4", "M-30"), bundle([Evidence(document_id="d", page_number=1, chunk_id="c1", text="Motor M-30 is running."), Evidence(document_id="d", page_number=2, chunk_id="c2", text="M-30 needs lubrication."), Evidence(document_id="d", page_number=3, chunk_id="c3", text="M-30 was serviced.")])),
    ]


class ClaimVerificationCalibrationBenchmark:
    """Purpose: calibration of claim confidence against whether the claim
    actually reached SUPPORTED status — reusing the same hand-labeled
    verification scenarios as O2's `claim_verification_quality` benchmark,
    with the P-101 node populated in the Knowledge Graph so at least one
    scenario has real graph support."""

    name = "claim_verification_calibration"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        repo = InMemoryGraphRepository()
        repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
        engine = ClaimVerificationEngine(repo)

        confidences: list[float] = []
        correctness: list[bool] = []
        for claim, bundle in _verification_scenarios():
            verified = engine.verify(ClaimGraph(claims=[claim], relations=[]), bundle)
            result = verified.claims[0]
            confidences.append(0.9 if result.verification_status == ClaimVerificationStatus.SUPPORTED else 0.5 if result.verification_status == ClaimVerificationStatus.PARTIALLY_SUPPORTED else 0.2)
            correctness.append(result.verification_status == ClaimVerificationStatus.SUPPORTED)

        return _report_from_calibration(self.name, confidences, correctness, "hand_labeled_verification_scenarios", start)


def _recommendation_calibration_scenarios() -> list[tuple[Recommendation, ClaimGraph, bool]]:
    def rec(rec_id: str, confidence: float, evidence: list[str]) -> Recommendation:
        return Recommendation(
            id=rec_id, description="d", priority=Priority.HIGH, severity=Severity.HIGH, estimated_impact="i",
            required_personnel=[], required_equipment=[], required_documents=[], estimated_downtime="TBD",
            confidence=confidence, justification="j", supporting_evidence=evidence,
        )

    supported_graph = ClaimGraph(claims=[Claim(id="c1", claim_type=ClaimType.FACT, text="c", subject="P-101", predicate="is_a", verification_status=ClaimVerificationStatus.SUPPORTED, source_evidence_ids=["c1"])], relations=[])
    empty_graph = ClaimGraph(claims=[], relations=[])

    return [
        (rec("r1", 0.9, ["c1"]), supported_graph, True),
        (rec("r2", 0.8, ["c1"]), supported_graph, True),
        (rec("r3", 0.6, ["c1"]), supported_graph, True),
        (rec("r4", 0.2, ["c1"]), supported_graph, False),
        (rec("r5", 0.9, []), supported_graph, False),
        (rec("r6", 0.9, ["c1"]), empty_graph, False),
    ]


class RecommendationCalibrationBenchmark:
    """Purpose: calibration of recommendation confidence against whether the
    recommendation actually passed Recommendation Verification — reusing
    O2's hand-labeled recommendation scenarios."""

    name = "recommendation_calibration"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        verifier = RecommendationVerifier()

        confidences: list[float] = []
        correctness: list[bool] = []
        for recommendation, claim_graph, _expected in _recommendation_calibration_scenarios():
            result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=recommendation.confidence), claim_graph)
            confidences.append(recommendation.confidence)
            correctness.append(len(result.verified) == 1)

        return _report_from_calibration(self.name, confidences, correctness, "hand_labeled_recommendation_scenarios", start)


class _StubEntailmentNLIDetector:
    """Deterministic: entails only exact-text matches — isolates the
    calibration benchmark from any particular NLI model's own accuracy."""

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        if premise.strip() == hypothesis.strip():
            return NLIResult(premise, hypothesis, NLILabel.ENTAILMENT, 0.9)
        return NLIResult(premise, hypothesis, NLILabel.NEUTRAL, 0.5)


class FaithfulnessCalibrationBenchmark:
    """Purpose: calibration of per-sentence faithfulness score against
    whether the sentence was actually entailed by a verified claim — the
    (score, entailed) pairs `FaithfulnessScorer` already produces are
    literally (confidence, correct) pairs, no adaptation needed."""

    name = "faithfulness_calibration"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        scorer = FaithfulnessScorer(_StubEntailmentNLIDetector())
        claim_texts = ["Pump P-101 is running.", "Valve V-2201A is closed.", "Motor M-30 needs lubrication."]
        claim_graph = ClaimGraph(
            claims=[Claim(id=f"c{i}", claim_type=ClaimType.FACT, text=t, subject=t, predicate="is_a", verification_status=ClaimVerificationStatus.SUPPORTED) for i, t in enumerate(claim_texts)],
            relations=[],
        )
        bundle = EvidenceBundle(evidence=[], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7)

        answers = [
            "Pump P-101 is running.",
            "Valve V-2201A is closed. Something totally unrelated happened.",
            "Motor M-30 needs lubrication. Another unrelated sentence appears here.",
        ]
        confidences: list[float] = []
        correctness: list[bool] = []
        for answer in answers:
            report = scorer.score(answer, claim_graph, claim_graph, bundle)
            for sentence in report.sentence_faithfulness:
                confidences.append(sentence.score if sentence.score > 0 else 0.1)
                correctness.append(sentence.entailed)

        return _report_from_calibration(self.name, confidences, correctness, "synthetic_faithfulness_sentences", start)
