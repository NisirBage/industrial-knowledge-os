from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import InMemoryGraphRepository
from app.retrieval.evidence import Evidence
from app.vision.defect.models import Defect, DefectType
from app.vision.graph.builder import VisionKnowledgeGraphBuilder
from app.vision.ingestion.models import VisionDocument, VisionSourceType
from app.vision.models import BoundingBox
from app.vision.ocr.models import OCRRegion
from app.vision.retrieval.multimodal import MultimodalRetriever


class _StubTextRetriever:
    def __init__(self, evidence: list[Evidence]):
        self._evidence = evidence

    def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
        return list(self._evidence)


def _graph_with_visual_evidence_for_p101():
    repo = InMemoryGraphRepository()
    repo.ensure_constraints()
    repo.create_node(EntityLabel.PUMP, canonical_name="P-101", confidence=1.0)

    builder = VisionKnowledgeGraphBuilder(repo)
    document = VisionDocument(
        id="doc-1", filename="p101.png", source_type=VisionSourceType.EQUIPMENT_PHOTO,
        content_type="image/png", width=300, height=300, storage_path="/tmp/p101.png", size_bytes=0, equipment_hint="P-101",
    )
    image_node_id = builder.add_image(document)
    builder.add_ocr_regions(image_node_id, [OCRRegion("P-101", 0.9, BoundingBox(0, 0, 10, 10), "doc-1")])
    builder.add_defects(image_node_id, [Defect(DefectType.CORROSION, BoundingBox(0, 0, 20, 20), 0.75, "doc-1", "medium")])
    return repo


def _retriever(graph_repository, text_evidence=None):
    entity_extraction_service = EntityExtractionService(RegexEntityExtractor())
    return MultimodalRetriever(_StubTextRetriever(text_evidence or []), graph_repository, entity_extraction_service)


def test_retrieve_merges_text_and_visual_evidence_into_one_list():
    repo = _graph_with_visual_evidence_for_p101()
    text_evidence = [Evidence(document_id="d1", page_number=1, chunk_id="c1", text="P-101 maintenance log", confidence=0.8)]
    retriever = _retriever(repo, text_evidence)

    results = retriever.retrieve("Why did P-101 fail?")

    assert any(e.chunk_id == "c1" for e in results)  # text evidence present
    assert any(e.source.startswith("vision:") for e in results)  # visual evidence present too


def test_visual_evidence_includes_ocr_and_defect_for_the_queried_equipment():
    repo = _graph_with_visual_evidence_for_p101()
    retriever = _retriever(repo)

    results = retriever.retrieve("What is the condition of P-101?")

    sources = {e.source for e in results}
    assert "vision:OCRRegion" in sources
    assert "vision:Defect" in sources


def test_visual_evidence_text_is_human_readable_not_a_raw_dict():
    repo = _graph_with_visual_evidence_for_p101()
    retriever = _retriever(repo)

    results = retriever.retrieve("What is the condition of P-101?")

    ocr_evidence = next(e for e in results if e.source == "vision:OCRRegion")
    assert "P-101" in ocr_evidence.text
    defect_evidence = next(e for e in results if e.source == "vision:Defect")
    assert "corrosion" in defect_evidence.text.lower()


def test_query_with_no_matching_equipment_returns_only_text_evidence():
    repo = _graph_with_visual_evidence_for_p101()
    text_evidence = [Evidence(document_id="d1", page_number=1, chunk_id="c1", text="general note", confidence=0.5)]
    retriever = _retriever(repo, text_evidence)

    results = retriever.retrieve("What is the weather today?")

    assert len(results) == 1
    assert results[0].chunk_id == "c1"


def test_visual_evidence_confidence_matches_the_underlying_node_confidence():
    repo = _graph_with_visual_evidence_for_p101()
    retriever = _retriever(repo)

    results = retriever.retrieve("What is the condition of P-101?")

    ocr_evidence = next(e for e in results if e.source == "vision:OCRRegion")
    assert ocr_evidence.confidence == 0.9


def test_text_retriever_failure_degrades_to_visual_evidence_only_not_a_crash():
    class _FailingTextRetriever:
        def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
            raise ValueError("Collection ikios_chunks not found")

    repo = _graph_with_visual_evidence_for_p101()
    entity_extraction_service = EntityExtractionService(RegexEntityExtractor())
    retriever = MultimodalRetriever(_FailingTextRetriever(), repo, entity_extraction_service)

    results = retriever.retrieve("What is the condition of P-101?")

    assert any(e.source.startswith("vision:") for e in results)


def test_no_duplicate_visual_evidence_across_multiple_query_entities():
    repo = _graph_with_visual_evidence_for_p101()
    retriever = _retriever(repo)

    # "P-101" appears twice conceptually via hint + direct mention; ensure no duplication.
    results = retriever.retrieve("Is P-101's P-101 gauge reading normal?")

    ocr_matches = [e for e in results if e.source == "vision:OCRRegion"]
    assert len(ocr_matches) == 1
