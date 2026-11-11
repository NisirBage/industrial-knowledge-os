"""Failure Intelligence Benchmarks (O3.4 Component 8).

These benchmarks build small synthetic `ExecutionTrace` fixtures with
KNOWN, constructed-by-hand ground truth (which failure categories exist,
which component is the "true" cause, which toggle should remove which
category) — the same hand-labeled-scenario pattern O1/O2's own benchmarks
already use (see `optimization_runners.py`/`calibration_runners.py`).
Because the ground truth is constructed rather than independently labeled,
precision/recall/accuracy here are self-consistency checks confirming the
real mining/attribution/counterfactual pipeline behaves as documented end
to end — not an external-validity claim about real-world diagnostic
accuracy (that would require a labeled production dataset this project
doesn't have).
"""

import time

from app.evaluation.models import NamedMetric
from app.execution.models import ExecutionEvent, ExecutionTrace
from app.failure_intelligence.attribution import FailureAttributionEngine
from app.failure_intelligence.clustering import RuleBasedClusteringStrategy
from app.failure_intelligence.counterfactual import CounterfactualSimulator, CounterfactualToggle
from app.failure_intelligence.miner import FailurePatternMiner
from app.failure_intelligence.optimization import OptimizationRecommendationEngine
from app.failure_intelligence.root_cause_graph import RootCauseGraphBuilder
from app.research.benchmarks.models import BenchmarkReport

_GROUND_TRUTH_CATEGORIES = {"retrieval_failure", "verification_failure"}


def _event(component: str, event_type: str, timestamp: str) -> ExecutionEvent:
    return ExecutionEvent(event_type=event_type, component=component, timestamp=timestamp, duration=0.05)


def _healthy_trace(execution_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        execution_id=execution_id, query=f"healthy query {execution_id}",
        events=[_event("planning", "planning_started", "2026-01-01T00:00:00+00:00"), _event("planning", "planning_finished", "2026-01-01T00:00:01+00:00")],
        confidence=0.9,
        retrieval_trace={"evidence_count": 3, "confidence": 0.8, "missing_information_count": 0},
        claim_extraction_trace={"claim_count": 2},
        verification_trace={"claim_coverage": 1.0, "unsupported_claim_count": 0},
    )


def _retrieval_failure_trace(execution_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        execution_id=execution_id, query=f"retrieval failure query {execution_id}",
        events=[_event("planning", "planning_started", "2026-01-01T00:00:00+00:00"), _event("retrieval", "retrieval_finished", "2026-01-01T00:00:01+00:00")],
        confidence=0.1, retrieval_trace=None,
    )


def _verification_failure_trace(execution_id: str) -> ExecutionTrace:
    return ExecutionTrace(
        execution_id=execution_id, query=f"verification failure query {execution_id}",
        events=[_event("planning", "planning_started", "2026-01-01T00:00:00+00:00"), _event("verification", "verification_finished", "2026-01-01T00:00:01+00:00")],
        confidence=0.3,
        retrieval_trace={"evidence_count": 2, "confidence": 0.7, "missing_information_count": 0},
        claim_extraction_trace={"claim_count": 3},
        verification_trace={"claim_coverage": 0.0, "unsupported_claim_count": 5},
    )


def _synthetic_traces() -> list[ExecutionTrace]:
    return [
        _healthy_trace("h1"), _healthy_trace("h2"), _healthy_trace("h3"),
        _retrieval_failure_trace("r1"), _retrieval_failure_trace("r2"),
        _verification_failure_trace("v1"), _verification_failure_trace("v2"),
    ]


class FailurePatternMiningBenchmark:
    """Pattern discovery precision/recall against constructed ground truth."""

    name = "failure_pattern_mining_quality"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        traces = _synthetic_traces()
        patterns = FailurePatternMiner(min_support=2).mine(traces)

        # Exclude the two cross-cutting/structural pattern categories
        # (they aren't per-subsystem failure categories comparable against
        # this ground truth set).
        mined_categories = {p.category for p in patterns} - {"repeated_failure_sequence", "confidence_collapse"}
        true_positives = mined_categories & _GROUND_TRUTH_CATEGORIES
        precision = round(len(true_positives) / len(mined_categories), 4) if mined_categories else 0.0
        recall = round(len(true_positives) / len(_GROUND_TRUTH_CATEGORIES), 4)

        metrics = [
            NamedMetric("pattern_discovery_precision", precision),
            NamedMetric("pattern_discovery_recall", recall),
            NamedMetric("patterns_mined", float(len(patterns))),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ground_truth_traces", len(traces))


class RootCauseAttributionBenchmark:
    """Root cause attribution accuracy: for the constructed retrieval-failure
    pattern, the true primary cause is HybridRetriever by construction
    (`ErrorAnalyzer._check_retrieval` always names it)."""

    name = "root_cause_attribution_accuracy"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        traces = [_retrieval_failure_trace("r1"), _retrieval_failure_trace("r2")]
        patterns = FailurePatternMiner(min_support=2).mine(traces)
        retrieval_pattern = next(p for p in patterns if p.category == "retrieval_failure")

        attribution = FailureAttributionEngine().attribute(retrieval_pattern, traces)
        correct = attribution.primary_cause == "HybridRetriever"

        metrics = [
            NamedMetric("attribution_accuracy", 1.0 if correct else 0.0),
            NamedMetric("evidence_confidence", attribution.evidence_confidence),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ground_truth_traces", len(traces))


class CounterfactualConsistencyBenchmark:
    """Counterfactual consistency: for each alarm-only toggle, disabling the
    corresponding check on a trace constructed to trigger it must remove
    that exact category — a self-consistency check on the deterministic
    transform model, not a claim about real causal effect."""

    name = "counterfactual_consistency"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        simulator = CounterfactualSimulator()

        cases = [
            (_verification_failure_trace("v1"), CounterfactualToggle.VERIFICATION, "verification_failure"),
            (ExecutionTrace(execution_id="n1", query="q", confidence=0.5, nli_trace={"contradiction_rate": 0.9}), CounterfactualToggle.NLI, "nli_failure"),
            (ExecutionTrace(execution_id="f1", query="q", confidence=0.5, faithfulness_trace={"overall_faithfulness": 0.1}), CounterfactualToggle.FAITHFULNESS, "faithfulness_failure"),
        ]

        consistent = 0
        for trace, toggle, expected_category in cases:
            result = simulator.simulate(trace, toggle)
            if expected_category in result.disappeared_categories:
                consistent += 1

        metrics = [NamedMetric("counterfactual_consistency", round(consistent / len(cases), 4))]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ground_truth_traces", len(cases))


class OptimizationRecommendationGroundednessBenchmark:
    """Every recommendation must cite real supporting evidence — a
    self-consistency quality gate on the (deterministic, template-driven)
    generator, verifying it never emits an unbacked recommendation."""

    name = "optimization_recommendation_groundedness"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        traces = _synthetic_traces()
        patterns = FailurePatternMiner(min_support=2).mine(traces)
        recommendations = OptimizationRecommendationEngine().generate(patterns=patterns)

        grounded = sum(1 for r in recommendations if r.evidence and r.supporting_execution_ids)
        precision = round(grounded / len(recommendations), 4) if recommendations else 1.0

        metrics = [
            NamedMetric("optimization_recommendation_precision", precision),
            NamedMetric("recommendations_generated", float(len(recommendations))),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ground_truth_traces", len(traces))


class FailureIntelligenceLatencyBenchmark:
    """Graph construction / clustering / counterfactual replay latency —
    real wall-clock measurements over the synthetic fixture set."""

    name = "failure_intelligence_latency"

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        traces = _synthetic_traces()

        graph_start = time.perf_counter()
        builder = RootCauseGraphBuilder()
        for trace in traces:
            builder.build(trace)
        graph_duration = time.perf_counter() - graph_start

        cluster_start = time.perf_counter()
        RuleBasedClusteringStrategy(key="category").cluster(traces)
        cluster_duration = time.perf_counter() - cluster_start

        replay_start = time.perf_counter()
        simulator = CounterfactualSimulator()
        for trace in traces:
            simulator.simulate(trace, CounterfactualToggle.VERIFICATION)
        replay_duration = time.perf_counter() - replay_start

        metrics = [
            NamedMetric("graph_construction_latency_seconds", round(graph_duration, 6)),
            NamedMetric("failure_clustering_latency_seconds", round(cluster_duration, 6)),
            NamedMetric("execution_replay_overhead_seconds", round(replay_duration, 6)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_ground_truth_traces", len(traces))
