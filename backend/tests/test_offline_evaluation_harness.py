from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import EntityLabel, RelationType
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.repositories.graph_repository import InMemoryGraphRepository
from app.research.datasets.generator import SyntheticDatasetGenerator
from app.research.offline_evaluation.harness import OfflineEvaluationHarness
from app.retrieval.evidence import Evidence


def _harness(graph_repository=None):
    return OfflineEvaluationHarness(
        entity_extraction_service=EntityExtractionService(RegexEntityExtractor()),
        relation_extractor=RuleBasedRelationExtractor(),
        graph_repository=graph_repository or InMemoryGraphRepository(),
        total_ontology_labels=len(EntityLabel),
        total_ontology_relations=len(RelationType),
    )


def _metric(report, name):
    return next((m.value for m in report.metrics if m.name == name), None)


async def test_core_metrics_always_present():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(5)
    report = await _harness().run(documents)

    metric_names = {m.name for m in report.metrics}
    assert "precision" in metric_names  # from entity extraction
    assert "node_count" in metric_names  # from knowledge graph


async def test_report_tracks_memory_and_cpu():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(3)
    report = await _harness().run(documents)

    assert report.memory_peak_bytes > 0
    assert report.cpu_time_seconds >= 0
    assert report.wall_time_seconds >= 0


class _FakeRetriever:
    def retrieve(self, query, top_k=8):
        return [Evidence(document_id="d1", page_number=1, chunk_id="c1", text="hit")]


async def test_optional_retrieval_section_activates_when_provided():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(2)
    report = await _harness().run(documents, retriever=_FakeRetriever(), retrieval_queries=[("q", {"c1"})])

    assert any(m.name == "recall_at_5" for m in report.metrics)


async def test_optional_retrieval_section_skipped_when_absent():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(2)
    report = await _harness().run(documents)

    assert not any(m.name == "recall_at_5" for m in report.metrics)


async def test_optional_faithfulness_section_activates_when_provided():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(2)
    report = await _harness().run(
        documents,
        faithfulness_answer="Pump P-101 requires lubrication",
        faithfulness_evidence_texts=["Pump P-101 requires lubrication per the manual"],
        faithfulness_cited_chunk_ids=["c1"],
        faithfulness_known_chunk_ids={"c1"},
    )

    assert _metric(report, "faithfulness") is not None


async def test_optional_reasoning_section_activates_when_provided():
    documents = SyntheticDatasetGenerator(seed=1).generate_dataset(2)
    from app.contradiction.detector import ContradictionDetector

    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-101 is open.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Valve V-101 is closed.", confidence=0.8),
    ]
    report = await _harness().run(
        documents, contradiction_detector=ContradictionDetector(), conflict_evidence=evidence, known_conflict_pairs=1
    )

    assert _metric(report, "contradictions_detected") == 1.0
