"""Failure Clustering Engine (O3.4 Component 2).

A pluggable `ClusteringStrategy` protocol with three real strategies:

- `RuleBasedClusteringStrategy`: deterministic exact-match grouping by
  failure category (via `ErrorAnalyzer.analyze_trace`), component sequence
  (via `reconstruct_timeline`), query intent (`planner_trace["task_type"]`,
  the same classification `TaskGraphPlanner` already computed), query
  entities/equipment (via `EntityExtractionService.extract`, the exact
  Chunk-wrapping pattern `PlannerAgent` already uses on query text), or
  timeline proximity (same time bucket by `started_at`).
- `EmbeddingBasedClusteringStrategy`: reuses a real `Embedder` to embed
  each trace's query and greedily clusters by cosine similarity.
- `GraphBasedClusteringStrategy`: reuses a real `GraphRepository` to
  resolve query entities to Knowledge Graph node ids, then clusters traces
  that share at least one node (graph neighborhood).

None of these strategies invent new telemetry — they read only
`trace.query` / already-derived trace signals, or call the exact same
production entity extraction / embedding / graph lookup every other
subsystem already uses.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

from app.documents.chunking import Chunk
from app.entity_extraction.service import EntityExtractionService
from app.error_analysis.analyzer import ErrorAnalyzer
from app.execution.models import ExecutionTrace
from app.execution.timeline import reconstruct_timeline


@dataclass(frozen=True)
class FailureCluster:
    cluster_id: str
    strategy_name: str
    key: str
    execution_ids: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.execution_ids)


class ClusteringStrategy(Protocol):
    name: str

    def cluster(self, traces: list[ExecutionTrace]) -> list[FailureCluster]: ...


def _query_chunk(query: str) -> Chunk:
    return Chunk(id="query", document_id="query", text=query, page_number=0, chunk_index=0)


class RuleBasedClusteringStrategy:
    """`key` selects the grouping dimension: "category" | "component_sequence"
    | "intent" | "entity" | "timeline_proximity"."""

    name = "rule_based"

    def __init__(self, key: str = "category", entity_extraction_service: EntityExtractionService | None = None, analyzer: ErrorAnalyzer | None = None):
        self._key = key
        self._entity_extraction_service = entity_extraction_service
        self._analyzer = analyzer or ErrorAnalyzer()

    def cluster(self, traces: list[ExecutionTrace]) -> list[FailureCluster]:
        groups: dict[str, list[str]] = {}
        for trace in traces:
            key = self._key_for(trace)
            if key is None:
                continue
            groups.setdefault(key, []).append(trace.execution_id)

        return [
            FailureCluster(cluster_id=f"{self.name}:{self._key}:{i}", strategy_name=self.name, key=key, execution_ids=execution_ids)
            for i, (key, execution_ids) in enumerate(groups.items())
        ]

    def _key_for(self, trace: ExecutionTrace) -> str | None:
        if self._key == "category":
            failures = self._analyzer.analyze_trace(trace).failures
            return "+".join(sorted({f.category.value for f in failures})) if failures else None
        if self._key == "component_sequence":
            sequence = reconstruct_timeline(trace).component_order
            return "->".join(sequence) if sequence else None
        if self._key == "intent":
            planner_trace = trace.planner_trace
            return planner_trace.get("task_type") if planner_trace else None
        if self._key == "entity":
            if self._entity_extraction_service is None:
                return None
            entities = self._entity_extraction_service.extract(_query_chunk(trace.query))
            return entities[0].canonical_name if entities else None
        if self._key == "timeline_proximity":
            started_at = trace.timestamps.get("started_at", "")
            # Bucket to the hour: "2026-01-01T14:23:00+00:00" -> "2026-01-01T14".
            return started_at[:13] if started_at else None
        raise ValueError(f"Unknown rule-based clustering key: {self._key!r}")


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingBasedClusteringStrategy:
    name = "embedding_based"

    def __init__(self, embedder, similarity_threshold: float = 0.85):
        self._embedder = embedder
        self._threshold = similarity_threshold

    def cluster(self, traces: list[ExecutionTrace]) -> list[FailureCluster]:
        if not traces:
            return []
        vectors = self._embedder.embed_texts([t.query for t in traces])

        assigned = [False] * len(traces)
        clusters: list[FailureCluster] = []
        for i, trace in enumerate(traces):
            if assigned[i]:
                continue
            execution_ids = [trace.execution_id]
            assigned[i] = True
            for j in range(i + 1, len(traces)):
                if assigned[j]:
                    continue
                if _cosine_similarity(vectors[i], vectors[j]) >= self._threshold:
                    execution_ids.append(traces[j].execution_id)
                    assigned[j] = True
            clusters.append(FailureCluster(cluster_id=f"{self.name}:{len(clusters)}", strategy_name=self.name, key=trace.query, execution_ids=execution_ids))
        return clusters


class GraphBasedClusteringStrategy:
    name = "graph_based"

    def __init__(self, entity_extraction_service: EntityExtractionService, graph_repository):
        self._entity_extraction_service = entity_extraction_service
        self._graph_repository = graph_repository

    def cluster(self, traces: list[ExecutionTrace]) -> list[FailureCluster]:
        node_to_executions: dict[str, list[str]] = {}
        for trace in traces:
            entities = self._entity_extraction_service.extract(_query_chunk(trace.query))
            for entity in entities:
                node = self._graph_repository.find_entity(entity.canonical_name, entity.label)
                if node is None:
                    continue
                node_to_executions.setdefault(node["id"], []).append(trace.execution_id)

        return [
            FailureCluster(cluster_id=f"{self.name}:{node_id}", strategy_name=self.name, key=node_id, execution_ids=sorted(set(execution_ids)))
            for node_id, execution_ids in node_to_executions.items()
            if len(set(execution_ids)) >= 2
        ]
