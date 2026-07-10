"""Execution Trace Benchmarks (O3.3 Component 9).

Measures real per-stage latency from actually running the orchestrator (the
same fakes-based construction `tests/test_orchestrator_integration.py` uses,
kept local to this file rather than importing test fixtures — the same
pattern `optimization_runners.py`'s `_synthetic_labeled_retriever_and_queries`
already establishes for benchmark-local fixtures), real cache effectiveness
from the actual `LRUTTLCache`, real event volume/trace size from an
assembled `ExecutionTrace`, and the real wall-clock cost the O3.3
instrumentation wrapper (`app.orchestrator.graph._timed`) itself adds over
an uninstrumented call — never a fabricated "instrumentation is free/costs
X%" number.
"""

import json
import time
from dataclasses import asdict

from app.agents.claim_extraction_agent import ClaimExtractionAgent
from app.agents.claim_verification_agent import ClaimVerificationAgent
from app.agents.confidence_propagation_agent import ConfidencePropagationAgent
from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.faithfulness_agent import FaithfulnessAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.recommendation_verification_agent import RecommendationVerificationAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.report_agent import ReportAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.caching.lru_cache import LRUTTLCache
from app.claims.extraction import ClaimExtractionEngine
from app.claims.propagation import GraphDecayConfidencePropagator
from app.claims.verification import ClaimVerificationEngine
from app.contradiction.detector import ContradictionDetector
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.models import NamedMetric
from app.evidence.fusion import EvidenceFusionEngine
from app.evidence.graph import EvidenceGraphBuilder
from app.execution.recorder import ExecutionRecorder
from app.faithfulness.scorer import FaithfulnessScorer
from app.narration.narrator import LLMNarrator
from app.nli.rule_based import RuleBasedNLIDetector
from app.orchestrator.graph import _timed, build_orchestrator
from app.planning.executor import TaskGraphExecutor
from app.planning.planner import TaskGraphPlanner
from app.reasoning.reasoning_graph_builder import ReasoningGraphBuilder
from app.recommendation.engine import RecommendationEngine
from app.recommendation.verification import RecommendationVerifier
from app.reflection.engine import SelfReflectionEngine
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.research.benchmarks.models import BenchmarkReport
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionStage
from app.services.llm_provider import MockLLMProvider

_BENCHMARK_QUERY = "Why has Pump P-101 repeatedly failed?"


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 8):
        return [
            Evidence(
                document_id="doc-1", page_number=3, chunk_id="c1",
                text="Pump P-101 experienced a bearing failure in March 2024 due to inadequate lubrication.",
                confidence=0.8, source="fusion",
            )
        ]


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def documents_mentioning(self, canonical_name: str):
        return [{"document_node_id": "doc-node-1", "document_id": "doc-1", "filename": "manual.pdf"}]

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        if relation_type is not None and relation_type.value == "USES_STANDARD":
            return []
        return [{"id": "node-failure1", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]

    def maintenance_history(self, canonical_name: str):
        return [{"id": "node-maint1", "type": "Inspection", "canonical_name": "Quarterly Inspection", "relation_type": "INSPECTED_BY"}]

    def incident_history(self, canonical_name: str):
        return [{"id": "node-incident1", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]


def _build_benchmark_orchestrator():
    llm_provider = MockLLMProvider()
    entity_extraction_service = EntityExtractionService(RegexEntityExtractor())
    graph_repository = _FakeGraphRepository()
    retriever = _FakeRetriever()

    task_graph_planner = TaskGraphPlanner()
    task_graph_executor = TaskGraphExecutor(retriever, graph_repository)
    graph_expansion = GraphExpansionStage(entity_extraction_service, graph_repository)
    nli_detector = RuleBasedNLIDetector()

    return build_orchestrator(
        planner_agent=PlannerAgent(entity_extraction_service, task_graph_planner),
        retrieval_agent=RetrievalAgent(task_graph_executor),
        knowledge_graph_agent=KnowledgeGraphAgent(graph_expansion),
        evidence_fusion_agent=EvidenceFusionAgent(EvidenceFusionEngine()),
        reasoning_agent=ReasoningAgent(ContradictionDetector(), EvidenceGraphBuilder(), ReasoningGraphBuilder()),
        claim_extraction_agent=ClaimExtractionAgent(ClaimExtractionEngine(entity_extraction_service, RuleBasedRelationExtractor())),
        claim_verification_agent=ClaimVerificationAgent(ClaimVerificationEngine(graph_repository)),
        confidence_propagation_agent=ConfidencePropagationAgent(GraphDecayConfidencePropagator()),
        recommendation_agent=RecommendationAgent(RecommendationEngine(llm_provider)),
        verifier_agent=VerifierAgent(),
        recommendation_verification_agent=RecommendationVerificationAgent(RecommendationVerifier()),
        report_agent=ReportAgent(LLMNarrator(llm_provider)),
        faithfulness_agent=FaithfulnessAgent(FaithfulnessScorer(nli_detector)),
        reflection_agent=ReflectionAgent(SelfReflectionEngine()),
    )


def _instrumentation_overhead_metrics(iterations: int = 500) -> list[NamedMetric]:
    def noop(state):
        return state

    start_raw = time.perf_counter()
    for _ in range(iterations):
        noop({"query": _BENCHMARK_QUERY})
    raw_duration = time.perf_counter() - start_raw

    wrapped = _timed("noop_stage", noop)
    start_wrapped = time.perf_counter()
    for _ in range(iterations):
        wrapped({"query": _BENCHMARK_QUERY})
    wrapped_duration = time.perf_counter() - start_wrapped

    overhead_per_call = (wrapped_duration - raw_duration) / iterations
    return [
        NamedMetric("instrumentation_overhead_per_call_seconds", round(overhead_per_call, 8)),
        NamedMetric("instrumentation_overhead_ratio", round(wrapped_duration / raw_duration, 4) if raw_duration > 0 else 0.0),
    ]


def _cache_effectiveness_metrics() -> list[NamedMetric]:
    cache: LRUTTLCache[str, str] = LRUTTLCache(max_size=10)
    keys = ["k1", "k2", "k3"]
    for key in keys:
        cache.set(key, key)
    for key in keys * 3:
        cache.get(key)
    cache.get("never-set")

    stats = cache.stats
    return [
        NamedMetric("cache_hit_ratio", stats.hit_ratio),
        NamedMetric("cache_size", float(stats.size)),
    ]


class ExecutionTraceBenchmark:
    """Real per-stage latency (from actually running the instrumented
    orchestrator), event volume/trace size, cache effectiveness, and
    instrumentation overhead — packed into one report since they're all
    facets of the same "what does O3.3 instrumentation cost/produce"
    question, matching how other benchmarks in this package (e.g.
    calibration) already pack several related metrics into one report."""

    name = "execution_trace_overhead"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()

        orchestrator = _build_benchmark_orchestrator()
        run_count = 3
        all_latencies: list[dict[str, float]] = []
        last_events = []
        for _ in range(run_count):
            final_state = orchestrator.invoke({"query": _BENCHMARK_QUERY})
            all_latencies.append(final_state["latencies"])
            last_events = final_state["execution_events"]

        stage_names = sorted(all_latencies[0].keys())
        stage_metrics = [
            NamedMetric(f"latency_{stage}_mean_seconds", round(sum(l[stage] for l in all_latencies) / run_count, 6))
            for stage in stage_names
        ]

        trace = ExecutionRecorder.finalize(
            execution_id="benchmark", query=_BENCHMARK_QUERY, events=last_events,
            latencies=all_latencies[-1], confidence=0.0, started_at="t0", completed_at="t1",
        )
        trace_size_bytes = len(json.dumps([asdict(e) for e in trace.events]))

        metrics = [
            *stage_metrics,
            NamedMetric("event_volume", float(len(last_events))),
            NamedMetric("trace_size_bytes", float(trace_size_bytes)),
            *_instrumentation_overhead_metrics(),
            *_cache_effectiveness_metrics(),
        ]

        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "orchestrator_fakes", run_count)
