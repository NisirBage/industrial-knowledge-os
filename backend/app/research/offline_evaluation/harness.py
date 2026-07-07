"""Offline Evaluation Harness (M5 Component 5): the single entry point that
runs every measurable subsystem against a dataset and produces one
consolidated report — callable from a plain script or a test, with no
FastAPI process required ("Support running evaluations without the web
frontend").

Core metrics (entity/relation extraction, knowledge graph coverage) always
run since they only need a document set. Retrieval, faithfulness,
recommendation-accuracy, and prediction-accuracy sections are optional —
each activates only when the caller supplies the inputs it needs (a query
set, an answer+evidence pair, accept/reject counts, or equipment ids with
known outcomes) — an evaluation that can't be measured honestly is skipped,
not filled with a placeholder number.
"""

import datetime as dt
import time
import tracemalloc
from dataclasses import dataclass, field

from app.contradiction.detector import ContradictionDetector
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.evaluators import FaithfulnessEvaluator
from app.evaluation.models import NamedMetric
from app.prediction.engine import PredictionEngine
from app.relation_extraction.base import RelationExtractor
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.research.benchmarks.runners import (
    EntityExtractionBenchmark,
    HybridRetrievalBenchmark,
    KnowledgeGraphBenchmark,
    PredictionBenchmark,
    ReasoningBenchmark,
    RelationExtractionBenchmark,
)
from app.research.datasets.models import SyntheticDocument
from app.retrieval.evidence import Evidence
from app.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True)
class OfflineEvaluationReport:
    metrics: list[NamedMetric]
    memory_current_bytes: int
    memory_peak_bytes: int
    cpu_time_seconds: float
    wall_time_seconds: float
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


class OfflineEvaluationHarness:
    def __init__(
        self,
        entity_extraction_service: EntityExtractionService,
        relation_extractor: RelationExtractor,
        graph_repository: GraphRepositoryProtocol,
        total_ontology_labels: int,
        total_ontology_relations: int,
    ):
        self._entity_extraction_service = entity_extraction_service
        self._relation_extractor = relation_extractor
        self._graph_repository = graph_repository
        self._total_ontology_labels = total_ontology_labels
        self._total_ontology_relations = total_ontology_relations

    async def run(
        self,
        documents: list[SyntheticDocument],
        retriever: HybridRetriever | None = None,
        retrieval_queries: list[tuple[str, set[str]]] | None = None,
        contradiction_detector: ContradictionDetector | None = None,
        conflict_evidence: list[Evidence] | None = None,
        known_conflict_pairs: int = 0,
        prediction_engine: PredictionEngine | None = None,
        prediction_equipment_ids: list[str] | None = None,
        prediction_actual_values: list[float] | None = None,
        faithfulness_answer: str | None = None,
        faithfulness_evidence_texts: list[str] | None = None,
        faithfulness_cited_chunk_ids: list[str] | None = None,
        faithfulness_known_chunk_ids: set[str] | None = None,
    ) -> OfflineEvaluationReport:
        tracemalloc.start()
        wall_start = time.perf_counter()
        cpu_start = time.process_time()

        metrics: list[NamedMetric] = []

        entity_report = await EntityExtractionBenchmark(self._entity_extraction_service, documents).run()
        metrics += entity_report.metrics

        relation_report = await RelationExtractionBenchmark(
            self._entity_extraction_service, self._relation_extractor, documents
        ).run()
        metrics += relation_report.metrics

        kg_report = await KnowledgeGraphBenchmark(
            self._graph_repository, self._total_ontology_labels, self._total_ontology_relations
        ).run()
        metrics += kg_report.metrics

        if retriever is not None and retrieval_queries:
            retrieval_report = await HybridRetrievalBenchmark(retriever, retrieval_queries).run()
            metrics += retrieval_report.metrics

        if contradiction_detector is not None and conflict_evidence is not None:
            reasoning_report = await ReasoningBenchmark(contradiction_detector, conflict_evidence, known_conflict_pairs).run()
            metrics += reasoning_report.metrics

        if prediction_engine is not None and prediction_equipment_ids and prediction_actual_values:
            prediction_report = await PredictionBenchmark(prediction_engine, prediction_equipment_ids, prediction_actual_values).run()
            metrics += prediction_report.metrics

        if faithfulness_answer is not None and faithfulness_evidence_texts is not None:
            metrics += FaithfulnessEvaluator().evaluate(
                faithfulness_answer,
                faithfulness_evidence_texts,
                faithfulness_cited_chunk_ids or [],
                faithfulness_known_chunk_ids or set(),
            )

        cpu_time = time.process_time() - cpu_start
        wall_time = time.perf_counter() - wall_start
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OfflineEvaluationReport(
            metrics=metrics,
            memory_current_bytes=current_bytes,
            memory_peak_bytes=peak_bytes,
            cpu_time_seconds=round(cpu_time, 4),
            wall_time_seconds=round(wall_time, 4),
        )
