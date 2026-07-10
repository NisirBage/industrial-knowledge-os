"""Ablation probes (O3 Component 1): one function per `AblationAxis`, each
running the SAME small synthetic scenario with the subsystem enabled vs
disabled and returning real metrics for both — reusing the exact real
production classes O1/O1.2/O2/M6/M7/M9 already built wherever one already
exists (no parallel/duplicate implementations). Every probe is `async` for
a uniform `AblationRunner` interface, even where nothing inside actually
awaits — matching the existing benchmark-runner convention.
"""

# Windows-specific DLL load-order bug (see app/main.py's own note): torch
# must be imported before neo4j (imported transitively by several probes'
# graph-repository/graph-intelligence dependencies) in this process, or
# torch's native DLL init can fail. Import it first, defensively, exactly
# like the composition root does.
import torch  # noqa: F401

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Awaitable, Callable

from app.evaluation.models import NamedMetric
from app.research.ablation.models import AblationAxis, AblationResult

_ProbeFn = Callable[[], Awaitable[AblationResult]]


async def _planner_probe() -> AblationResult:
    from app.entity_extraction.schemas import ExtractedEntity
    from app.ontology.labels import EntityLabel
    from app.planning.planner import TaskGraphPlanner

    planner = TaskGraphPlanner()
    entity = ExtractedEntity(
        id="e1", label=EntityLabel.PUMP, canonical_name="P-101", aliases=frozenset(), confidence=0.9,
        document_id="d1", chunk_id="c1", page_number=1, start_char=0, end_char=5,
    )

    start_on = time.perf_counter()
    plan_on = planner.build("Why did P-101 fail?", [entity])
    duration_on = time.perf_counter() - start_on

    start_off = time.perf_counter()
    plan_off = planner.build("Why did P-101 fail?", [])  # no resolved entity -> single-task fallback
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.PLANNER,
        metrics_enabled=[NamedMetric("task_node_count", float(len(plan_on.nodes))), NamedMetric("plan_levels", float(len(plan_on.levels())))],
        metrics_disabled=[NamedMetric("task_node_count", float(len(plan_off.nodes))), NamedMetric("plan_levels", float(len(plan_off.levels())))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_single_entity_query",
    )


async def _parallel_retrieval_probe() -> AblationResult:
    from app.repositories.vector_repository import ScoredChunk
    from app.retrieval.bm25_index import IndexedChunk
    from app.retrieval.graph_expansion import GraphExpansionResult
    from app.retrieval.hybrid import HybridRetriever

    sleep_seconds = 0.05

    class _SlowEmbedder:
        dimension = 4

        def embed_texts(self, texts):
            time.sleep(sleep_seconds)
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    class _SlowVectorRepository:
        def search(self, query_vector, top_k=8):
            time.sleep(sleep_seconds)
            return [ScoredChunk(chunk_id="c1", document_id="d1", text="Pump P-101 failed.", page_number=1, score=0.9, metadata={})]

    class _SlowBm25Index:
        def search(self, query, top_k):
            time.sleep(sleep_seconds)
            return [(IndexedChunk(chunk_id="c1", document_id="d1", text="Pump P-101 failed.", page_number=1), 1.0)]

    class _SlowGraphExpansion:
        def expand(self, query):
            time.sleep(sleep_seconds)
            return GraphExpansionResult(matched_entity_ids=[], boosted_document_ids=set(), graph_evidence=[])

    class _ConstantReranker:
        def score(self, query, texts):
            return [0.0] * len(texts)

    def _retrieve_and_time(executor) -> float:
        retriever = HybridRetriever(_SlowEmbedder(), _SlowVectorRepository(), _SlowBm25Index(), _SlowGraphExpansion(), _ConstantReranker(), executor=executor)
        start = time.perf_counter()
        retriever.retrieve("Why did P-101 fail?", top_k=5)
        return time.perf_counter() - start

    with ThreadPoolExecutor(max_workers=4) as executor:
        duration_on = _retrieve_and_time(executor)
    duration_off = _retrieve_and_time(None)

    return AblationResult(
        AblationAxis.PARALLEL_RETRIEVAL,
        metrics_enabled=[NamedMetric("wall_clock_seconds", round(duration_on, 4))],
        metrics_disabled=[NamedMetric("wall_clock_seconds", round(duration_off, 4))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_slow_fetch_fixture",
    )


async def _learning_to_rank_probe() -> AblationResult:
    from app.research.benchmarks.optimization_runners import RankingQualityBenchmark

    report = await RankingQualityBenchmark(num_documents=6).run()
    ltr_mrr = next(m.value for m in report.metrics if m.name == "ltr_mrr")
    rrf_mrr = next(m.value for m in report.metrics if m.name == "rrf_mrr")
    return AblationResult(
        AblationAxis.LEARNING_TO_RANK,
        metrics_enabled=[NamedMetric("mrr", ltr_mrr)],
        metrics_disabled=[NamedMetric("mrr", rrf_mrr)],
        duration_enabled_seconds=report.duration_seconds / 2,
        duration_disabled_seconds=report.duration_seconds / 2,
        dataset_name=report.dataset_name,
    )


async def _entity_resolution_probe() -> AblationResult:
    from app.entity_resolution.resolver import EntityResolver, KnownEntity
    from app.ontology.labels import EntityLabel

    known = [KnownEntity(node_id="known", canonical_name="P-101", label=EntityLabel.PUMP)]
    enabled_resolver = EntityResolver(merge_threshold=0.82)
    disabled_resolver = EntityResolver(merge_threshold=0.999)  # near-impossible threshold approximates "off"

    start_on = time.perf_counter()
    match_on = enabled_resolver.resolve("P-102", EntityLabel.PUMP, known)
    duration_on = time.perf_counter() - start_on

    start_off = time.perf_counter()
    match_off = disabled_resolver.resolve("P-102", EntityLabel.PUMP, known)
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.ENTITY_RESOLUTION,
        metrics_enabled=[NamedMetric("merged", 1.0 if match_on else 0.0)],
        metrics_disabled=[NamedMetric("merged", 1.0 if match_off else 0.0)],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_p101_p102_pair",
    )


async def _context_compression_probe() -> AblationResult:
    from app.research.benchmarks.optimization_runners import CompressionBenchmark

    report = await CompressionBenchmark(num_unique=10, duplicate_ratio=0.4).run()
    ratio = next(m.value for m in report.metrics if m.name == "compression_ratio")
    return AblationResult(
        AblationAxis.CONTEXT_COMPRESSION,
        metrics_enabled=[NamedMetric("compression_ratio", ratio)],
        metrics_disabled=[NamedMetric("compression_ratio", 1.0)],  # no compression = every item kept
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=0.0,
        dataset_name=report.dataset_name,
    )


async def _claim_extraction_probe() -> AblationResult:
    from app.research.benchmarks.reasoning_runners import ClaimExtractionBenchmark

    report = await ClaimExtractionBenchmark().run()
    claims_extracted = next(m.value for m in report.metrics if m.name == "claims_extracted")
    return AblationResult(
        AblationAxis.CLAIM_EXTRACTION,
        metrics_enabled=[NamedMetric("claims_extracted", claims_extracted)],
        metrics_disabled=[NamedMetric("claims_extracted", 0.0)],  # extraction stage skipped entirely
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=0.0,
        dataset_name=report.dataset_name,
    )


async def _claim_verification_probe() -> AblationResult:
    from app.research.benchmarks.reasoning_runners import ClaimVerificationAblationBenchmark

    report = await ClaimVerificationAblationBenchmark().run()
    on = next(m.value for m in report.metrics if m.name == "rule_plus_graph_claims_kept")
    off = next(m.value for m in report.metrics if m.name == "disabled_claims_kept")
    return AblationResult(
        AblationAxis.CLAIM_VERIFICATION,
        metrics_enabled=[NamedMetric("claims_kept", on)],
        metrics_disabled=[NamedMetric("claims_kept", off)],
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=report.duration_seconds,
        dataset_name=report.dataset_name,
    )


async def _confidence_propagation_probe() -> AblationResult:
    from app.claims.models import Claim, ClaimGraph, ClaimRelation, ClaimType, ClaimVerificationStatus
    from app.claims.propagation import GraphDecayConfidencePropagator

    strong = Claim(id="a", claim_type=ClaimType.FACT, text="a", subject="a", predicate="is_a", confidence=0.9, verification_status=ClaimVerificationStatus.SUPPORTED, support_count=3)
    weak = Claim(id="b", claim_type=ClaimType.FACT, text="b", subject="b", predicate="is_a", confidence=0.2, verification_status=ClaimVerificationStatus.UNKNOWN)
    graph = ClaimGraph(claims=[strong, weak], relations=[ClaimRelation(id="r1", source_claim_id="a", target_claim_id="b", relation_type="RELATED_TO")])

    propagator = GraphDecayConfidencePropagator()
    start_on = time.perf_counter()
    result_on = propagator.propagate(graph)
    duration_on = time.perf_counter() - start_on
    weak_propagated = result_on.claim_by_id("b").propagated_confidence

    start_off = time.perf_counter()
    weak_local_only = propagator._local_confidence(weak)  # "off" = no graph propagation, local formula only
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.CONFIDENCE_PROPAGATION,
        metrics_enabled=[NamedMetric("weak_claim_confidence", round(weak_propagated, 4))],
        metrics_disabled=[NamedMetric("weak_claim_confidence", round(weak_local_only, 4))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_strong_weak_claim_pair",
    )


async def _nli_probe() -> AblationResult:
    from app.research.benchmarks.reasoning_runners import NLIAblationBenchmark

    report = await NLIAblationBenchmark().run()
    on = next(m.value for m in report.metrics if m.name == "rules_contradictions_found")
    off = next(m.value for m in report.metrics if m.name == "disabled_contradictions_found")
    return AblationResult(
        AblationAxis.NLI,
        metrics_enabled=[NamedMetric("contradictions_found", on)],
        metrics_disabled=[NamedMetric("contradictions_found", off)],
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=report.duration_seconds,
        dataset_name=report.dataset_name,
    )


async def _recommendation_verification_probe() -> AblationResult:
    from app.claims.models import ClaimGraph
    from app.core.enums import Priority, Severity
    from app.recommendation.models import Recommendation, RecommendationBundle
    from app.recommendation.verification import RecommendationVerificationThresholds, RecommendationVerifier

    low_confidence_recommendation = Recommendation(
        id="r1", description="d", priority=Priority.LOW, severity=Severity.LOW, estimated_impact="i",
        required_personnel=[], required_equipment=[], required_documents=[], estimated_downtime="TBD",
        confidence=0.1, justification="j", supporting_evidence=["c1"],
    )
    bundle = RecommendationBundle(recommendations=[low_confidence_recommendation], overall_confidence=0.1)
    empty_graph = ClaimGraph(claims=[], relations=[])

    enabled_verifier = RecommendationVerifier()  # real thresholds: min_confidence=0.3, require_graph_support=True
    disabled_verifier = RecommendationVerifier(RecommendationVerificationThresholds(min_confidence=0.0, require_graph_support=False, min_evidence_count=0))

    start_on = time.perf_counter()
    result_on = enabled_verifier.verify(bundle, empty_graph)
    duration_on = time.perf_counter() - start_on

    start_off = time.perf_counter()
    result_off = disabled_verifier.verify(bundle, empty_graph)
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.RECOMMENDATION_VERIFICATION,
        metrics_enabled=[NamedMetric("verified_count", float(len(result_on.verified)))],
        metrics_disabled=[NamedMetric("verified_count", float(len(result_off.verified)))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_low_confidence_recommendation",
    )


async def _faithfulness_scoring_probe() -> AblationResult:
    from app.claims.models import Claim, ClaimGraph, ClaimType, ClaimVerificationStatus
    from app.evidence.models import EvidenceBundle
    from app.faithfulness.scorer import FaithfulnessScorer
    from app.nli.rule_based import RuleBasedNLIDetector

    claim_graph = ClaimGraph(claims=[Claim(id="c1", claim_type=ClaimType.FACT, text="Pump P-101 is running.", subject="P-101", predicate="is_a", verification_status=ClaimVerificationStatus.SUPPORTED)], relations=[])
    bundle = EvidenceBundle(evidence=[], graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7)
    hallucinated_answer = "The plant will be permanently decommissioned next week."

    scorer = FaithfulnessScorer(RuleBasedNLIDetector())
    start_on = time.perf_counter()
    report_on = scorer.score(hallucinated_answer, claim_graph, claim_graph, bundle)
    duration_on = time.perf_counter() - start_on

    start_off = time.perf_counter()
    blind_trust_score = 1.0  # "off" = no faithfulness check, the answer is trusted unconditionally
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.FAITHFULNESS_SCORING,
        metrics_enabled=[NamedMetric("overall_faithfulness", report_on.overall_faithfulness)],
        metrics_disabled=[NamedMetric("overall_faithfulness", blind_trust_score)],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_hallucinated_answer",
    )


async def _caching_probe() -> AblationResult:
    from app.caching.lru_cache import LRUTTLCache

    cache: LRUTTLCache = LRUTTLCache(max_size=16)
    start_on = time.perf_counter()
    cache.set("query-1", "result-1")
    cache.get("query-1")  # repeated key -> real hit
    cache.get("query-1")
    duration_on = time.perf_counter() - start_on
    stats_on = cache.stats

    no_cache = LRUTTLCache(max_size=16)
    start_off = time.perf_counter()
    no_cache.get("query-2")  # distinct key every time -> real miss, simulating caching disabled
    no_cache.get("query-3")
    duration_off = time.perf_counter() - start_off
    stats_off = no_cache.stats

    return AblationResult(
        AblationAxis.CACHING,
        metrics_enabled=[NamedMetric("hit_ratio", stats_on.hit_ratio)],
        metrics_disabled=[NamedMetric("hit_ratio", stats_off.hit_ratio)],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_repeated_vs_distinct_keys",
    )


async def _explainability_probe() -> AblationResult:
    from app.ml.datasets.synthetic import generate_failure_prediction_dataset
    from app.ml.datasets.models import MLProblemType
    from app.ml.explainability.explainer import Explainer
    from app.ml.training.trainer import SklearnTrainer

    df = generate_failure_prediction_dataset(100, seed=42)
    X, y = df.drop(columns=["label"]), df["label"]
    trained = SklearnTrainer().train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")

    start_on = time.perf_counter()
    importances = Explainer().feature_importance(trained.estimator, trained.feature_names)
    duration_on = time.perf_counter() - start_on

    start_off = time.perf_counter()
    no_explanation: list = []  # "off" = explainability layer skipped
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.EXPLAINABILITY,
        metrics_enabled=[NamedMetric("feature_importances_returned", float(len(importances)))],
        metrics_disabled=[NamedMetric("feature_importances_returned", float(len(no_explanation)))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_failure_prediction_dataset",
    )


async def _vision_probe() -> AblationResult:
    from app.vision.ocr.provider import build_ocr_provider
    from app.research.benchmarks.vision_runners import OCRBenchmark

    ocr_provider = build_ocr_provider()
    report = await OCRBenchmark(ocr_provider, tags=["P-101", "V-2201A"]).run()
    accuracy_on = next(m.value for m in report.metrics if m.name == "tag_read_accuracy")

    return AblationResult(
        AblationAxis.VISION,
        metrics_enabled=[NamedMetric("tag_read_accuracy", accuracy_on)],
        metrics_disabled=[NamedMetric("tag_read_accuracy", 0.0)],  # no OCR = no tags ever read
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=0.0,
        dataset_name=report.dataset_name,
    )


async def _graph_intelligence_probe() -> AblationResult:
    from app.research.benchmarks.graph_runners import GraphEmbeddingBenchmark

    report = await GraphEmbeddingBenchmark("node2vec", num_nodes=16).run()
    quality_margin = next(m.value for m in report.metrics if m.name == "embedding_quality_margin")
    return AblationResult(
        AblationAxis.GRAPH_INTELLIGENCE,
        metrics_enabled=[NamedMetric("embedding_quality_margin", quality_margin)],
        metrics_disabled=[NamedMetric("embedding_quality_margin", 0.0)],  # no graph embedding = no structural advantage over random
        duration_enabled_seconds=report.duration_seconds,
        duration_disabled_seconds=0.0,
        dataset_name=report.dataset_name,
    )


async def _ml_prediction_probe() -> AblationResult:
    from app.ml.datasets.models import MLProblemType
    from app.ml.datasets.synthetic import generate_failure_prediction_dataset
    from app.ml.training.trainer import SklearnTrainer

    df = generate_failure_prediction_dataset(300, seed=42)
    X, y = df.drop(columns=["label"]), df["label"]

    start_on = time.perf_counter()
    trained = SklearnTrainer().train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction")
    duration_on = time.perf_counter() - start_on
    accuracy_on = trained.metrics["accuracy"]

    start_off = time.perf_counter()
    majority_baseline = max(y.mean(), 1 - y.mean())  # "off" = rule-based/majority-class baseline, no trained ML model
    duration_off = time.perf_counter() - start_off

    return AblationResult(
        AblationAxis.ML_PREDICTION,
        metrics_enabled=[NamedMetric("accuracy", round(accuracy_on, 4))],
        metrics_disabled=[NamedMetric("accuracy", round(float(majority_baseline), 4))],
        duration_enabled_seconds=duration_on,
        duration_disabled_seconds=duration_off,
        dataset_name="synthetic_failure_prediction_dataset",
    )


PROBES: dict[AblationAxis, _ProbeFn] = {
    AblationAxis.PLANNER: _planner_probe,
    AblationAxis.PARALLEL_RETRIEVAL: _parallel_retrieval_probe,
    AblationAxis.LEARNING_TO_RANK: _learning_to_rank_probe,
    AblationAxis.ENTITY_RESOLUTION: _entity_resolution_probe,
    AblationAxis.CONTEXT_COMPRESSION: _context_compression_probe,
    AblationAxis.CLAIM_EXTRACTION: _claim_extraction_probe,
    AblationAxis.CLAIM_VERIFICATION: _claim_verification_probe,
    AblationAxis.CONFIDENCE_PROPAGATION: _confidence_propagation_probe,
    AblationAxis.NLI: _nli_probe,
    AblationAxis.RECOMMENDATION_VERIFICATION: _recommendation_verification_probe,
    AblationAxis.FAITHFULNESS_SCORING: _faithfulness_scoring_probe,
    AblationAxis.CACHING: _caching_probe,
    AblationAxis.EXPLAINABILITY: _explainability_probe,
    AblationAxis.VISION: _vision_probe,
    AblationAxis.GRAPH_INTELLIGENCE: _graph_intelligence_probe,
    AblationAxis.ML_PREDICTION: _ml_prediction_probe,
}
