"""Failure Pattern Mining & Root Cause Intelligence API (O3.4): read/compute
endpoints over the `ExecutionTrace`s `SqlExecutionRepository` already
persists — mining, clustering, root cause graph export, counterfactual
simulation, and optimization recommendations. Entirely read-side / on-demand
compute; nothing here is on the chat request path, and nothing here
modifies production settings (counterfactual simulation and optimization
recommendations are strictly advisory, per this milestone's constraint).
"""

from fastapi import APIRouter, HTTPException, Request

from app.entity_extraction.service import EntityExtractionService
from app.execution.models import ExecutionTrace
from app.failure_intelligence.attribution import FailureAttributionEngine
from app.failure_intelligence.clustering import (
    EmbeddingBasedClusteringStrategy,
    GraphBasedClusteringStrategy,
    RuleBasedClusteringStrategy,
)
from app.failure_intelligence.counterfactual import CounterfactualSimulator, CounterfactualToggle
from app.failure_intelligence.miner import FailurePatternMiner
from app.failure_intelligence.models import FailurePattern
from app.failure_intelligence.optimization import OptimizationRecommendationEngine
from app.failure_intelligence.root_cause_graph import RootCauseGraph, RootCauseGraphBuilder
from app.schemas.failure_intelligence import (
    CounterfactualRequest,
    CounterfactualResultOut,
    FailureAttributionOut,
    FailureClusterOut,
    FailurePatternOut,
    OptimizationRecommendationOut,
    RootCauseEdgeOut,
    RootCauseGraphOut,
    RootCauseNodeOut,
)

router = APIRouter(prefix="/failure-intelligence", tags=["failure-intelligence"])


def _pattern_to_out(pattern: FailurePattern) -> FailurePatternOut:
    return FailurePatternOut(
        pattern_id=pattern.pattern_id, category=pattern.category, support=pattern.support, frequency=pattern.frequency,
        severity=pattern.severity.value, affected_components=pattern.affected_components,
        example_execution_ids=pattern.example_execution_ids, confidence=pattern.confidence, description=pattern.description,
    )


def _graph_to_out(graph: RootCauseGraph) -> RootCauseGraphOut:
    exported = graph.to_export_dict()
    return RootCauseGraphOut(
        nodes=[RootCauseNodeOut(**n) for n in exported["nodes"]],
        edges=[RootCauseEdgeOut(**e) for e in exported["edges"]],
    )


@router.post("/patterns/mine", response_model=list[FailurePatternOut])
async def mine_patterns(request: Request, limit: int = 100) -> list[FailurePatternOut]:
    execution_repository = request.app.state.execution_repository
    pattern_repository = request.app.state.failure_pattern_repository

    traces = await execution_repository.list_traces(limit=limit)
    patterns = FailurePatternMiner().mine(traces)
    await pattern_repository.save_all(patterns)
    return [_pattern_to_out(p) for p in patterns]


@router.get("/patterns", response_model=list[FailurePatternOut])
async def list_patterns(request: Request) -> list[FailurePatternOut]:
    pattern_repository = request.app.state.failure_pattern_repository
    return [_pattern_to_out(p) for p in await pattern_repository.list_all()]


@router.get("/patterns/{pattern_id}/attribution", response_model=FailureAttributionOut)
async def get_pattern_attribution(pattern_id: str, request: Request) -> FailureAttributionOut:
    pattern_repository = request.app.state.failure_pattern_repository
    execution_repository = request.app.state.execution_repository

    pattern = await pattern_repository.get(pattern_id)
    if pattern is None:
        raise HTTPException(status_code=404, detail=f"No failure pattern with id {pattern_id!r}")

    traces = [t for t in (await execution_repository.get_trace(eid) for eid in pattern.example_execution_ids) if t is not None]
    attribution = FailureAttributionEngine().attribute(pattern, traces)

    return FailureAttributionOut(
        pattern_id=attribution.pattern_id, primary_cause=attribution.primary_cause, secondary_cause=attribution.secondary_cause,
        contributing_components=attribution.contributing_components, evidence_confidence=attribution.evidence_confidence,
        execution_confidence=attribution.execution_confidence, severity=attribution.severity.value,
        historical_frequency=attribution.historical_frequency,
    )


@router.get("/clusters", response_model=list[FailureClusterOut])
async def get_clusters(request: Request, strategy: str = "rule_based", key: str = "category", limit: int = 100) -> list[FailureClusterOut]:
    execution_repository = request.app.state.execution_repository
    traces = await execution_repository.list_traces(limit=limit)

    if strategy == "rule_based":
        entity_extraction_service: EntityExtractionService = request.app.state.entity_extraction_service
        clusters = RuleBasedClusteringStrategy(key=key, entity_extraction_service=entity_extraction_service).cluster(traces)
    elif strategy == "embedding_based":
        clusters = EmbeddingBasedClusteringStrategy(request.app.state.embedder).cluster(traces)
    elif strategy == "graph_based":
        clusters = GraphBasedClusteringStrategy(request.app.state.entity_extraction_service, request.app.state.graph_repository).cluster(traces)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown clustering strategy {strategy!r}. Valid: rule_based, embedding_based, graph_based")

    return [FailureClusterOut(cluster_id=c.cluster_id, strategy_name=c.strategy_name, key=c.key, execution_ids=c.execution_ids, size=c.size) for c in clusters]


@router.get("/root-cause-graph/{execution_id}", response_model=RootCauseGraphOut)
async def get_root_cause_graph(execution_id: str, request: Request) -> RootCauseGraphOut:
    execution_repository = request.app.state.execution_repository
    pattern_repository = request.app.state.failure_pattern_repository

    trace = await execution_repository.get_trace(execution_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No execution trace with id {execution_id!r}")

    patterns = await pattern_repository.list_all()
    graph = RootCauseGraphBuilder().build(trace, patterns=patterns)
    return _graph_to_out(graph)


@router.post("/counterfactual/{execution_id}", response_model=CounterfactualResultOut)
async def simulate_counterfactual(execution_id: str, payload: CounterfactualRequest, request: Request) -> CounterfactualResultOut:
    execution_repository = request.app.state.execution_repository
    trace: ExecutionTrace | None = await execution_repository.get_trace(execution_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"No execution trace with id {execution_id!r}")

    try:
        toggle = CounterfactualToggle(payload.toggle)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown toggle {payload.toggle!r}. Valid: {[t.value for t in CounterfactualToggle]}") from exc

    result = CounterfactualSimulator().simulate(trace, toggle)
    return CounterfactualResultOut(
        execution_id=result.execution_id, toggle=result.toggle, applicable=result.applicable,
        baseline_failure_categories=result.baseline_failure_categories, simulated_failure_categories=result.simulated_failure_categories,
        disappeared_categories=result.disappeared_categories, confidence=result.confidence,
        expected_improvement=result.expected_improvement, note=result.note,
    )


@router.get("/optimization-recommendations", response_model=list[OptimizationRecommendationOut])
async def get_optimization_recommendations(request: Request) -> list[OptimizationRecommendationOut]:
    pattern_repository = request.app.state.failure_pattern_repository
    patterns = await pattern_repository.list_all()
    recommendations = OptimizationRecommendationEngine().generate(patterns=patterns)

    return [
        OptimizationRecommendationOut(
            subsystem=r.subsystem, explanation=r.explanation, evidence=r.evidence,
            supporting_execution_ids=r.supporting_execution_ids, expected_benefit=r.expected_benefit, confidence=r.confidence,
        )
        for r in recommendations
    ]
