from app.embeddings.embedder import HashingEmbedder
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.execution.models import ExecutionEvent, ExecutionTrace
from app.failure_intelligence.clustering import (
    EmbeddingBasedClusteringStrategy,
    GraphBasedClusteringStrategy,
    RuleBasedClusteringStrategy,
)
from app.repositories.graph_repository import InMemoryGraphRepository


def _trace(execution_id: str, query: str, started_at: str = "2026-01-01T10:00:00+00:00", **subsystem_traces) -> ExecutionTrace:
    events = [
        ExecutionEvent(event_type="planning_started", component="planning", timestamp=started_at, duration=0.0),
        ExecutionEvent(event_type="planning_finished", component="planning", timestamp=started_at, duration=0.1),
    ]
    return ExecutionTrace(execution_id=execution_id, query=query, events=events, timestamps={"started_at": started_at}, **subsystem_traces)


def test_rule_based_clustering_by_category_groups_matching_failures():
    strategy = RuleBasedClusteringStrategy(key="category")
    traces = [
        _trace("e1", "q1", retrieval_trace=None),
        _trace("e2", "q2", retrieval_trace=None),
        _trace("e3", "q3", retrieval_trace={"evidence_count": 2, "confidence": 0.8, "missing_information_count": 0}, claim_extraction_trace={"claim_count": 1}),
    ]

    clusters = strategy.cluster(traces)
    retrieval_cluster = next(c for c in clusters if c.key == "retrieval_failure")
    assert set(retrieval_cluster.execution_ids) == {"e1", "e2"}


def test_rule_based_clustering_by_component_sequence():
    strategy = RuleBasedClusteringStrategy(key="component_sequence")
    traces = [_trace("e1", "q1"), _trace("e2", "q2")]

    clusters = strategy.cluster(traces)
    assert len(clusters) == 1
    assert clusters[0].key == "planning"
    assert set(clusters[0].execution_ids) == {"e1", "e2"}


def test_rule_based_clustering_by_intent():
    strategy = RuleBasedClusteringStrategy(key="intent")
    traces = [
        _trace("e1", "q1", planner_trace={"node_count": 1, "failed_node_count": 0, "task_type": "general_retrieval"}),
        _trace("e2", "q2", planner_trace={"node_count": 1, "failed_node_count": 0, "task_type": "general_retrieval"}),
        _trace("e3", "q3", planner_trace={"node_count": 1, "failed_node_count": 0, "task_type": "failure_analysis"}),
    ]

    clusters = strategy.cluster(traces)
    general = next(c for c in clusters if c.key == "general_retrieval")
    assert set(general.execution_ids) == {"e1", "e2"}


def test_rule_based_clustering_by_entity_reuses_real_extraction_service():
    service = EntityExtractionService(RegexEntityExtractor())
    strategy = RuleBasedClusteringStrategy(key="entity", entity_extraction_service=service)
    traces = [
        _trace("e1", "Why did Pump P-101 fail?"),
        _trace("e2", "What caused Pump P-101 to fail again?"),
        _trace("e3", "Nothing recognizable here."),
    ]

    clusters = strategy.cluster(traces)
    assert any(set(c.execution_ids) == {"e1", "e2"} for c in clusters)


def test_rule_based_clustering_by_timeline_proximity_buckets_by_hour():
    strategy = RuleBasedClusteringStrategy(key="timeline_proximity")
    traces = [
        _trace("e1", "q1", started_at="2026-01-01T10:15:00+00:00"),
        _trace("e2", "q2", started_at="2026-01-01T10:45:00+00:00"),
        _trace("e3", "q3", started_at="2026-01-01T14:00:00+00:00"),
    ]

    clusters = strategy.cluster(traces)
    hour_10_cluster = next(c for c in clusters if set(c.execution_ids) == {"e1", "e2"})
    assert hour_10_cluster.key == "2026-01-01T10"


def test_rule_based_clustering_rejects_unknown_key():
    strategy = RuleBasedClusteringStrategy(key="not_a_real_key")
    try:
        strategy.cluster([_trace("e1", "q1")])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_embedding_based_clustering_groups_identical_queries():
    strategy = EmbeddingBasedClusteringStrategy(HashingEmbedder(), similarity_threshold=0.99)
    traces = [
        _trace("e1", "pump bearing failure"),
        _trace("e2", "pump bearing failure"),
        _trace("e3", "completely different valve issue text"),
    ]

    clusters = strategy.cluster(traces)
    assert any(set(c.execution_ids) == {"e1", "e2"} for c in clusters)
    assert any(c.execution_ids == ["e3"] for c in clusters)


def test_embedding_based_clustering_handles_empty_trace_list():
    assert EmbeddingBasedClusteringStrategy(HashingEmbedder()).cluster([]) == []


def test_graph_based_clustering_groups_traces_sharing_a_graph_node():
    from app.ontology.labels import EntityLabel

    graph_repository = InMemoryGraphRepository()
    graph_repository.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    service = EntityExtractionService(RegexEntityExtractor())
    strategy = GraphBasedClusteringStrategy(service, graph_repository)

    traces = [
        _trace("e1", "Why did Pump P-101 fail?"),
        _trace("e2", "What caused Pump P-101 to overheat?"),
        _trace("e3", "Unrelated query with no known entity."),
    ]

    clusters = strategy.cluster(traces)
    assert len(clusters) == 1
    assert set(clusters[0].execution_ids) == {"e1", "e2"}


def test_graph_based_clustering_excludes_singleton_groups():
    from app.ontology.labels import EntityLabel

    graph_repository = InMemoryGraphRepository()
    graph_repository.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    service = EntityExtractionService(RegexEntityExtractor())
    strategy = GraphBasedClusteringStrategy(service, graph_repository)

    clusters = strategy.cluster([_trace("e1", "Why did Pump P-101 fail?")])
    assert clusters == []
