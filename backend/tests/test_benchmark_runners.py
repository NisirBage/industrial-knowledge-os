from app.contradiction.detector import ContradictionDetector
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import EntityLabel, RelationType
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.repositories.graph_repository import InMemoryGraphRepository
from app.research.benchmarks.runners import (
    EntityExtractionBenchmark,
    HybridRetrievalBenchmark,
    KnowledgeGraphBenchmark,
    PredictionBenchmark,
    ReasoningBenchmark,
    RecommendationBenchmark,
    RelationExtractionBenchmark,
    TimelineBenchmark,
)
from app.research.datasets.generator import SyntheticDatasetGenerator
from app.retrieval.evidence import Evidence
from app.timeline.engine import TimelineEngine


def _metric(report, name):
    return next(m.value for m in report.metrics if m.name == name)


async def test_entity_extraction_benchmark_scores_perfect_extraction_highly():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(10, doc_types=["maintenance_log"])
    service = EntityExtractionService(RegexEntityExtractor())

    report = await EntityExtractionBenchmark(service, documents).run()

    assert report.sample_size == 10
    assert _metric(report, "f1") > 0.5  # regex should catch equipment tags reliably


async def test_relation_extraction_benchmark_produces_report():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(10, doc_types=["incident_report"])
    service = EntityExtractionService(RegexEntityExtractor())
    relation_extractor = RuleBasedRelationExtractor()

    report = await RelationExtractionBenchmark(service, relation_extractor, documents).run()

    assert report.benchmark_name == "relation_extraction"
    assert any(m.name == "f1" for m in report.metrics)


async def test_knowledge_graph_benchmark_reports_coverage():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101")
    repo.create_node(EntityLabel.VALVE, "V-1")
    repo.create_relation(
        repo.find_entity("P-101")["id"], repo.find_entity("V-1")["id"], RelationType.CONNECTED_TO
    )

    report = await KnowledgeGraphBenchmark(repo, total_ontology_labels=len(EntityLabel), total_ontology_relations=len(RelationType)).run()

    assert _metric(report, "node_count") == 2
    assert _metric(report, "edge_count") == 1
    assert 0 < _metric(report, "ontology_entity_coverage") < 1


class _FakeRetriever:
    def __init__(self, hit_map: dict[str, list[str]]):
        self._hit_map = hit_map

    def retrieve(self, query: str, top_k: int = 8):
        return [Evidence(document_id="d1", page_number=1, chunk_id=cid, text="text") for cid in self._hit_map.get(query, [])]


async def test_hybrid_retrieval_benchmark_perfect_recall():
    retriever = _FakeRetriever({"q1": ["c1", "c2"]})
    queries = [("q1", {"c1", "c2"})]

    report = await HybridRetrievalBenchmark(retriever, queries, top_k=2).run()

    assert _metric(report, "recall_at_2") == 1.0


async def test_reasoning_benchmark_measures_contradiction_detection_recall():
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-101 is closed.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Valve V-101 is open.", confidence=0.8),
    ]

    report = await ReasoningBenchmark(ContradictionDetector(), evidence, known_conflict_pairs=1).run()

    assert _metric(report, "recall") == 1.0


class _FakeLearningRepositoryForBenchmark:
    async def feedback_volume(self) -> int:
        return 5


async def test_recommendation_benchmark_reports_feedback_volume():
    report = await RecommendationBenchmark(_FakeLearningRepositoryForBenchmark()).run()
    assert _metric(report, "recommendation_feedback_volume") == 5.0


class _FakeGraphRepositoryForTimeline:
    def find_entity(self, canonical_name, label=None):
        return {"id": "node-1", "type": "Pump", "canonical_name": canonical_name, "aliases": []}

    def find_neighbors(self, entity_id, relation_type=None, depth=1):
        return [{"id": "n1", "type": "Inspection", "canonical_name": "Inspection A", "relation_type": "INSPECTED_BY"}]

    def maintenance_history(self, canonical_name):
        return []

    def incident_history(self, canonical_name):
        return []


async def test_timeline_benchmark_reports_determinism():
    engine = TimelineEngine(_FakeGraphRepositoryForTimeline())
    report = await TimelineBenchmark(engine, "P-101").run()

    assert _metric(report, "is_deterministic") == 1.0
    assert _metric(report, "event_count") == 1.0


class _FakePrediction:
    def __init__(self, value: float):
        self.value = value


class _FakePredictionEngine:
    async def predict(self, equipment_id, task_type):
        return _FakePrediction(value=0.5)


async def test_prediction_benchmark_reports_error_against_actuals():
    report = await PredictionBenchmark(_FakePredictionEngine(), ["P-101", "P-102"], [0.5, 0.5]).run()

    assert _metric(report, "mean_absolute_error") == 0.0
    assert _metric(report, "prediction_accuracy") == 1.0
