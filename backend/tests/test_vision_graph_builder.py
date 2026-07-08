from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository
from app.vision.defect.models import Defect, DefectType
from app.vision.detection.models import Detection
from app.vision.diagram.models import DiagramSymbol, SymbolType
from app.vision.gauge.models import GaugeReading
from app.vision.graph.builder import VisionKnowledgeGraphBuilder
from app.vision.ingestion.models import VisionDocument, VisionSourceType
from app.vision.models import BoundingBox
from app.vision.ocr.models import OCRRegion


def _repo() -> InMemoryGraphRepository:
    repo = InMemoryGraphRepository()
    repo.ensure_constraints()
    return repo


def _document(**overrides) -> VisionDocument:
    base = dict(
        id="doc-1", filename="gauge.png", source_type=VisionSourceType.EQUIPMENT_PHOTO,
        content_type="image/png", width=300, height=300, storage_path="/tmp/gauge.png", size_bytes=1024, equipment_hint=None,
    )
    base.update(overrides)
    return VisionDocument(**base)


def test_add_image_creates_image_node():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)

    image_node_id = builder.add_image(_document())

    assert repo.node_count() == 1
    assert "Image" in repo.distinct_entity_types()


def test_add_image_links_depicts_relation_when_equipment_hint_resolves():
    repo = _repo()
    repo.create_node(EntityLabel.PUMP, canonical_name="P-101", confidence=1.0)
    builder = VisionKnowledgeGraphBuilder(repo)

    image_node_id = builder.add_image(_document(equipment_hint="P-101"))

    neighbors = repo.find_neighbors(image_node_id, relation_type=RelationType.DEPICTS)
    assert any(n["canonical_name"] == "P-101" for n in neighbors)


def test_add_ocr_regions_creates_nodes_and_read_from_relation():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    region = OCRRegion(text="P-101", confidence=0.95, bounding_box=BoundingBox(0, 0, 50, 20), image_id="doc-1")
    node_ids = builder.add_ocr_regions(image_node_id, [region])

    assert len(node_ids) == 1
    assert "OCRRegion" in repo.distinct_entity_types()
    neighbors = repo.find_neighbors(node_ids[0], relation_type=RelationType.READ_FROM)
    assert any(n["id"] == image_node_id for n in neighbors)


def test_add_diagram_symbols_links_associated_with_ontology_entity():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    symbol = DiagramSymbol(symbol_type=SymbolType.VALVE, bounding_box=BoundingBox(0, 0, 40, 40), confidence=0.6, image_id="doc-1")
    node_ids = builder.add_diagram_symbols(image_node_id, [symbol])

    assert "DiagramSymbol" in repo.distinct_entity_types()
    associated = repo.find_neighbors(node_ids[0], relation_type=RelationType.ASSOCIATED_WITH)
    assert any(n["type"] == "Valve" for n in associated)


def test_add_detections_links_associated_with_when_equipment_type_known():
    repo = _repo()
    repo.create_node(EntityLabel.PUMP, canonical_name="P-101", confidence=1.0)
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    detection = Detection(equipment_type="P-101", bounding_box=BoundingBox(0, 0, 50, 50), confidence=0.7, image_reference="doc-1")
    node_ids = builder.add_detections(image_node_id, [detection])

    associated = repo.find_neighbors(node_ids[0], relation_type=RelationType.ASSOCIATED_WITH)
    assert any(n["canonical_name"] == "P-101" for n in associated)


def test_add_detections_with_unknown_equipment_type_has_no_associated_with():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    detection = Detection(equipment_type="unknown", bounding_box=BoundingBox(0, 0, 50, 50), confidence=0.7, image_reference="doc-1")
    node_ids = builder.add_detections(image_node_id, [detection])

    associated = repo.find_neighbors(node_ids[0], relation_type=RelationType.ASSOCIATED_WITH)
    assert associated == []


def test_add_gauge_readings_creates_nodes_and_relation():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    reading = GaugeReading(bounding_box=BoundingBox(0, 0, 60, 60), image_id="doc-1", needle_angle_degrees=0.0, normalized_value=0.5, confidence=0.6, valid=True, validation_note="OK")
    node_ids = builder.add_gauge_readings(image_node_id, [reading])

    assert "GaugeReading" in repo.distinct_entity_types()
    neighbors = repo.find_neighbors(node_ids[0], relation_type=RelationType.READ_FROM)
    assert any(n["id"] == image_node_id for n in neighbors)


def test_add_defects_creates_nodes_and_detected_in_relation():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    defect = Defect(defect_type=DefectType.CORROSION, bounding_box=BoundingBox(0, 0, 30, 30), confidence=0.8, image_id="doc-1", severity="medium")
    node_ids = builder.add_defects(image_node_id, [defect])

    assert "Defect" in repo.distinct_entity_types()
    neighbors = repo.find_neighbors(node_ids[0], relation_type=RelationType.DETECTED_IN)
    assert any(n["id"] == image_node_id for n in neighbors)


def test_full_pipeline_produces_a_connected_multimodal_graph():
    repo = _repo()
    builder = VisionKnowledgeGraphBuilder(repo)
    image_node_id = builder.add_image(_document())

    builder.add_ocr_regions(image_node_id, [OCRRegion("P-101", 0.9, BoundingBox(0, 0, 10, 10), "doc-1")])
    builder.add_defects(image_node_id, [Defect(DefectType.CORROSION, BoundingBox(0, 0, 10, 10), 0.7, "doc-1", "low")])

    assert repo.node_count() == 3  # image + ocr region + defect
    assert repo.edge_count() == 2  # each linked back to the image
