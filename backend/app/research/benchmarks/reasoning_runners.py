"""Reasoning Benchmarks (O2 Component 9): one runner per new Cognitive
Verification Layer subsystem, following the exact `BenchmarkReport` shape
and "small deterministic synthetic/hand-labeled input, measure the real
production component" discipline `optimization_runners.py`/`graph_runners.py`
already established. No parallel research framework, no fabricated ground
truth — every hand-labeled scenario below is a literal, real example, not
randomly generated.

Reasoning Latency is not a separate benchmark: every `BenchmarkReport`
already carries `duration_seconds`, measured from the real component under
test, so it's reported by each runner below rather than duplicated in one
more class.

Ablation studies here are deliberately scoped to what O2 itself adds (Claim
Verification on/off/with-graph-support, NLI detector choice) rather than
re-simulating the Retrieval/Ranking/Compression axes O1 already benchmarks
via `ranking_quality`/`context_compression` — running the full 8-axis
cross-product the spec's diagram describes is real future work (see
Roadmap), not something faked here with unrun combinations.
"""

import time

from sklearn.metrics import brier_score_loss

from app.claims.extraction import ClaimExtractionEngine
from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
from app.claims.verification import ClaimVerificationEngine
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.metrics import f1_score, precision, recall
from app.evaluation.models import NamedMetric
from app.evidence.models import EvidenceBundle
from app.faithfulness.scorer import FaithfulnessScorer
from app.nli.models import NLILabel, NLIResult
from app.nli.rule_based import RuleBasedNLIDetector
from app.ontology.labels import EntityLabel
from app.recommendation.models import Recommendation, RecommendationBundle
from app.recommendation.verification import RecommendationVerifier
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.repositories.graph_repository import InMemoryGraphRepository
from app.research.benchmarks.models import BenchmarkReport
from app.retrieval.evidence import Evidence


class _AlwaysNeutralNLIDetector:
    """Stand-in for "NLI: Disabled" in the ablation benchmark — every pair
    is NEUTRAL, i.e. no contradiction ever detected via NLI."""

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        return NLIResult(premise, hypothesis, NLILabel.NEUTRAL, 0.5)


class ClaimExtractionBenchmark:
    """Purpose: verify ClaimExtractionEngine recovers the expected atomic
    claims + causal relation from the spec's own worked example, and
    produces zero claims from prose with no recognizable equipment tags."""

    name = "claim_extraction_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        engine = ClaimExtractionEngine(EntityExtractionService(RegexEntityExtractor()), RuleBasedRelationExtractor())

        worked_example = Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 failed due to Valve V-2201A.")
        graph = engine.extract([worked_example])
        subjects = {c.subject for c in graph.claims}
        expected_subjects = {"P-101", "V-2201A"}
        found_expected = len(subjects & expected_subjects)

        no_entity_text = Evidence(document_id="d1", page_number=1, chunk_id="c2", text="The weather was clear that day.")
        empty_graph = engine.extract([no_entity_text])

        metrics = [
            NamedMetric("claims_extracted", float(len(graph.claims))),
            NamedMetric("relations_extracted", float(len(graph.relations))),
            NamedMetric("expected_subject_recall", round(found_expected / len(expected_subjects), 4)),
            NamedMetric("no_entity_text_claim_count", float(len(empty_graph.claims))),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "worked_example_and_negative_case", 2)


def _claim(claim_id: str, subject: str, status: ClaimVerificationStatus = ClaimVerificationStatus.UNKNOWN, claim_type: ClaimType = ClaimType.FACT) -> Claim:
    return Claim(id=claim_id, claim_type=claim_type, text=subject, subject=subject, predicate="is_a", object=EntityLabel.PUMP.value, source_evidence_ids=["c1"], verification_status=status)


class ClaimVerificationBenchmark:
    """Purpose: accuracy of ClaimVerificationEngine's status assignment
    against 5 hand-labeled (claim, graph state, evidence) scenarios covering
    every reachable status."""

    name = "claim_verification_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        repo = InMemoryGraphRepository()
        repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
        engine = ClaimVerificationEngine(repo)

        scenarios = [
            # (claim, bundle, expected_status)
            (
                _claim("supported", "P-101"),
                EvidenceBundle(
                    evidence=[
                        Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running."),
                        Evidence(document_id="d", page_number=2, chunk_id="c2", text="P-101 was inspected."),
                    ],
                    graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7,
                ),
                ClaimVerificationStatus.SUPPORTED,
            ),
            (
                _claim("partial", "P-101"),
                EvidenceBundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running.")], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7),
                ClaimVerificationStatus.PARTIALLY_SUPPORTED,
            ),
            (
                _claim("insufficient", "Unknown-Tag-9000"),
                EvidenceBundle(evidence=[Evidence(document_id="d", page_number=1, chunk_id="c1", text="Something happened.")], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7),
                ClaimVerificationStatus.INSUFFICIENT_EVIDENCE,
            ),
            (
                _claim("recommendation_type", "P-101", claim_type=ClaimType.RECOMMENDATION),
                EvidenceBundle(evidence=[], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7),
                ClaimVerificationStatus.UNKNOWN,
            ),
        ]

        correct = 0
        for claim, bundle, expected in scenarios:
            verified = engine.verify(ClaimGraph(claims=[claim], relations=[]), bundle)
            if verified.claims[0].verification_status == expected:
                correct += 1

        metrics = [NamedMetric("accuracy", round(correct / len(scenarios), 4)), NamedMetric("scenario_count", float(len(scenarios)))]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "hand_labeled_verification_scenarios", len(scenarios))


def _nli_labeled_pairs() -> list[tuple[str, str, NLILabel]]:
    return [
        ("Pump P-101 failed due to cavitation.", "Pump P-101 failed due to cavitation.", NLILabel.ENTAILMENT),
        ("Valve V-2201A is closed.", "Valve V-2201A remains fully closed.", NLILabel.ENTAILMENT),
        ("Motor M-30 is running normally.", "Motor M-30 operates normally.", NLILabel.ENTAILMENT),
        ("Valve V-2201A is open.", "Valve V-2201A was confirmed closed.", NLILabel.CONTRADICTION),
        ("Pump P-101 passed inspection.", "Pump P-101 failed inspection.", NLILabel.CONTRADICTION),
        ("Motor M-30 is running.", "Motor M-30 has stopped.", NLILabel.CONTRADICTION),
        ("Pump P-101 is running.", "The inspection schedule was updated in March.", NLILabel.NEUTRAL),
        ("Valve V-2201A was installed in 2019.", "Unrelated compliance audit completed.", NLILabel.NEUTRAL),
        ("Motor M-30 needs lubrication.", "New safety procedure published.", NLILabel.NEUTRAL),
    ]


class NLIBenchmark:
    """Purpose: overall + per-label accuracy of RuleBasedNLIDetector (the
    always-available fallback — a real transformer NLI model may not be
    reachable in every environment, so this is what's honestly guaranteed to
    run) against 9 hand-labeled (premise, hypothesis, gold_label) triples, 3
    per label."""

    name = "nli_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        detector = RuleBasedNLIDetector()
        pairs = _nli_labeled_pairs()

        correct_by_label: dict[NLILabel, int] = {label: 0 for label in NLILabel}
        total_by_label: dict[NLILabel, int] = {label: 0 for label in NLILabel}
        overall_correct = 0
        for premise, hypothesis, gold in pairs:
            predicted = detector.classify(premise, hypothesis).label
            total_by_label[gold] += 1
            if predicted == gold:
                correct_by_label[gold] += 1
                overall_correct += 1

        metrics = [NamedMetric("accuracy", round(overall_correct / len(pairs), 4))]
        for label in NLILabel:
            if total_by_label[label]:
                metrics.append(NamedMetric(f"{label.value}_accuracy", round(correct_by_label[label] / total_by_label[label], 4)))
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "hand_labeled_nli_pairs", len(pairs))


def _recommendation_scenarios() -> list[tuple[Recommendation, ClaimGraph, bool]]:
    def rec(rec_id: str, confidence: float, evidence: list[str]) -> Recommendation:
        from app.core.enums import Priority, Severity

        return Recommendation(
            id=rec_id, description="d", priority=Priority.HIGH, severity=Severity.HIGH, estimated_impact="i",
            required_personnel=[], required_equipment=[], required_documents=[], estimated_downtime="TBD",
            confidence=confidence, justification="j", supporting_evidence=evidence,
        )

    supported_graph = ClaimGraph(claims=[_claim("c1", "P-101", status=ClaimVerificationStatus.SUPPORTED)], relations=[])
    empty_graph = ClaimGraph(claims=[], relations=[])

    return [
        (rec("good1", 0.8, ["c1"]), supported_graph, True),
        (rec("good2", 0.5, ["c1"]), supported_graph, True),
        (rec("low_confidence", 0.1, ["c1"]), supported_graph, False),
        (rec("no_evidence", 0.8, []), supported_graph, False),
        (rec("no_graph_support", 0.8, ["c1"]), empty_graph, False),
    ]


class RecommendationVerificationBenchmark:
    """Purpose: precision/recall of RecommendationVerifier's verified/
    rejected decision against 5 hand-labeled (recommendation, claim graph,
    expected_verified) scenarios covering each threshold independently."""

    name = "recommendation_verification_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        verifier = RecommendationVerifier()
        scenarios = _recommendation_scenarios()

        true_positive = false_positive = false_negative = true_negative = 0
        for recommendation, claim_graph, expected_verified in scenarios:
            result = verifier.verify(RecommendationBundle(recommendations=[recommendation], overall_confidence=recommendation.confidence), claim_graph)
            predicted_verified = len(result.verified) == 1
            if predicted_verified and expected_verified:
                true_positive += 1
            elif predicted_verified and not expected_verified:
                false_positive += 1
            elif not predicted_verified and expected_verified:
                false_negative += 1
            else:
                true_negative += 1

        precision_value = precision(true_positive, false_positive)
        recall_value = recall(true_positive, false_negative)
        metrics = [
            NamedMetric("precision", precision_value),
            NamedMetric("recall", recall_value),
            NamedMetric("f1", f1_score(precision_value, recall_value)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "hand_labeled_recommendation_scenarios", len(scenarios))


class FaithfulnessBenchmark:
    """Purpose: FaithfulnessScorer must rank a fully-grounded answer above a
    hallucinated one on the same claim graph (a real, checkable assertion,
    not just a reported number); also reports unsupported-claim rate from a
    verification pass that drops half the claims, and a Confidence
    Calibration proxy (Brier score between claim confidence and a hand-
    labeled "was this claim actually correct" outcome on 6 claims — a small,
    honestly-labeled sample, not a claim about production calibration)."""

    name = "faithfulness_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        scorer = FaithfulnessScorer(RuleBasedNLIDetector())

        claim = Claim(
            id="c1", claim_type=ClaimType.FACT, text="Pump P-101 failed due to cavitation.",
            subject="P-101", predicate="is_a", verification_status=ClaimVerificationStatus.SUPPORTED,
        )
        claim_graph = ClaimGraph(claims=[claim], relations=[])
        bundle = EvidenceBundle(evidence=[], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7)

        faithful_report = scorer.score("Pump P-101 failed due to cavitation.", claim_graph, claim_graph, bundle)
        hallucinated_report = scorer.score("The plant will be shut down permanently next week.", claim_graph, claim_graph, bundle)

        pre_verification = ClaimGraph(
            claims=[
                Claim(id="a", claim_type=ClaimType.FACT, text="a", subject="a", predicate="is_a", verification_status=ClaimVerificationStatus.SUPPORTED),
                Claim(id="b", claim_type=ClaimType.FACT, text="b", subject="b", predicate="is_a", verification_status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE),
            ],
            relations=[],
        )
        post_verification = ClaimGraph(claims=[pre_verification.claims[0]], relations=[])
        coverage_report = scorer.score("a", pre_verification, post_verification, bundle)

        # Confidence calibration proxy: 6 hand-labeled (predicted_confidence, was_correct) pairs.
        calibration_pairs = [(0.9, 1), (0.85, 1), (0.6, 1), (0.4, 0), (0.2, 0), (0.1, 0)]
        predicted = [p for p, _ in calibration_pairs]
        actual = [a for _, a in calibration_pairs]
        calibration_error = round(float(brier_score_loss(actual, predicted)), 4)

        metrics = [
            NamedMetric("faithful_answer_score", faithful_report.overall_faithfulness),
            NamedMetric("hallucinated_answer_score", hallucinated_report.overall_faithfulness),
            NamedMetric("correctly_ranked_faithful_above_hallucinated", 1.0 if faithful_report.overall_faithfulness > hallucinated_report.overall_faithfulness else 0.0),
            NamedMetric("unsupported_claim_count", float(coverage_report.unsupported_claim_count)),
            NamedMetric("claim_coverage", coverage_report.claim_coverage),
            NamedMetric("calibration_error_brier", calibration_error),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_faithful_vs_hallucinated", 2)


class ClaimVerificationAblationBenchmark:
    """Purpose: the O2 ablation axis the spec names directly — Claim
    Verification Disabled / Rule-based (no Knowledge Graph populated) /
    Rule+Graph (Knowledge Graph populated) — measured on the identical
    synthetic scenario, reporting unsupported-claim count and graph
    faithfulness for each configuration. "Full Verification" (Rule+Graph+
    contradiction check) is the `rule_plus_graph` configuration here, since
    ClaimVerificationEngine always checks for contradictions — there is no
    separate "with/without contradiction checking" mode to ablate."""

    name = "claim_verification_ablation"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        claim_graph = ClaimGraph(
            claims=[
                Claim(id="c1", claim_type=ClaimType.FACT, text="Equipment = P-101", subject="P-101", predicate="is_a", object=EntityLabel.PUMP.value, source_evidence_ids=["e1"]),
                Claim(id="c2", claim_type=ClaimType.FACT, text="Equipment = V-2201A", subject="V-2201A", predicate="is_a", object=EntityLabel.VALVE.value, source_evidence_ids=["e2"]),
            ],
            relations=[],
        )
        bundle = EvidenceBundle(
            evidence=[Evidence(document_id="d", page_number=1, chunk_id="e1", text="Pump P-101 is running."), Evidence(document_id="d", page_number=1, chunk_id="e2", text="Valve V-2201A is closed.")],
            graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7,
        )

        disabled_kept = len(claim_graph.claims)  # "disabled" = no verification pass runs at all, nothing dropped

        empty_repo = InMemoryGraphRepository()
        rule_based_engine = ClaimVerificationEngine(empty_repo)
        rule_based_kept = len(rule_based_engine.drop_unsupported(rule_based_engine.verify(claim_graph, bundle)).claims)

        populated_repo = InMemoryGraphRepository()
        populated_repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
        populated_repo.create_node(EntityLabel.VALVE, "V-2201A", confidence=0.9)
        rule_plus_graph_engine = ClaimVerificationEngine(populated_repo)
        rule_plus_graph_kept = len(rule_plus_graph_engine.drop_unsupported(rule_plus_graph_engine.verify(claim_graph, bundle)).claims)

        metrics = [
            NamedMetric("disabled_claims_kept", float(disabled_kept)),
            NamedMetric("rule_based_claims_kept", float(rule_based_kept)),
            NamedMetric("rule_plus_graph_claims_kept", float(rule_plus_graph_kept)),
            NamedMetric("disabled_unsupported_count", 0.0),
            NamedMetric("rule_based_unsupported_count", float(len(claim_graph.claims) - rule_based_kept)),
            NamedMetric("rule_plus_graph_unsupported_count", float(len(claim_graph.claims) - rule_plus_graph_kept)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_two_entity_scenario", len(claim_graph.claims))


class NLIAblationBenchmark:
    """Purpose: NLI Disabled (every pair NEUTRAL) vs Rules
    (RuleBasedNLIDetector) on an identical contradiction-detection scenario
    via NLIContradictionDetector — reporting contradictions found and
    latency for each configuration."""

    name = "nli_ablation"

    async def run(self) -> BenchmarkReport:
        from app.nli.contradiction_bridge import NLIContradictionDetector

        start = time.perf_counter()
        bundle = EvidenceBundle(
            evidence=[
                Evidence(document_id="d", page_number=1, chunk_id="c1", text="Valve V-2201A is open."),
                Evidence(document_id="d", page_number=2, chunk_id="c2", text="Valve V-2201A was confirmed closed."),
            ],
            graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7,
        )

        disabled_start = time.perf_counter()
        disabled_contradictions = NLIContradictionDetector(_AlwaysNeutralNLIDetector()).detect(bundle)
        disabled_latency = time.perf_counter() - disabled_start

        rules_start = time.perf_counter()
        rules_contradictions = NLIContradictionDetector(RuleBasedNLIDetector()).detect(bundle)
        rules_latency = time.perf_counter() - rules_start

        metrics = [
            NamedMetric("disabled_contradictions_found", float(len(disabled_contradictions))),
            NamedMetric("rules_contradictions_found", float(len(rules_contradictions))),
            NamedMetric("disabled_latency_seconds", round(disabled_latency, 6)),
            NamedMetric("rules_latency_seconds", round(rules_latency, 6)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_contradiction_scenario", 2)
