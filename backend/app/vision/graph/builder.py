"""Vision Knowledge Graph Builder (M7 Component 7): the image analog of
KnowledgeGraphBuilder — every vision analysis output becomes a graph node,
never an isolated blob returned only to the caller. Reuses the exact same
GraphRepositoryProtocol (Neo4j or in-memory) every other subsystem writes
to, so vision nodes/edges are queryable alongside text-derived ones with no
separate storage.

Node canonical names: instance-level vision nodes (an OCR hit, a specific
detection, a specific defect...) are not canonical named domain entities the
way "Pump P-101" is — two different corrosion patches in two different
photos are not "the same entity." Each gets a fresh UUID as its
canonical_name, matching how Document nodes already use their own uuid
document_id rather than a human name.
"""

import uuid

from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.vision.defect.models import Defect
from app.vision.detection.models import Detection
from app.vision.diagram.models import DiagramSymbol
from app.vision.diagram.parser import ontology_label_for_symbol
from app.vision.gauge.models import GaugeReading
from app.vision.ingestion.models import VisionDocument
from app.vision.ocr.models import OCRRegion


def _node_properties(confidence: float, image_node_id: str, bounding_box, extra: dict) -> dict:
    """Every vision node's full property set includes its own confidence and
    a back-reference to its source image node id — duplicated here from the
    first-class `confidence` field `create_node` already stores, since
    `find_neighbors()` doesn't project node confidence into its return shape
    and widening that shared, widely-used method's contract wasn't worth the
    blast radius for this one new consumer. `get_node_properties()` is the
    single place any caller (the Multimodal Retriever included) needs to
    look to fully describe a vision node."""
    return {
        "confidence": confidence,
        "image_node_id": image_node_id,
        "x_min": bounding_box.x_min,
        "y_min": bounding_box.y_min,
        "x_max": bounding_box.x_max,
        "y_max": bounding_box.y_max,
        **extra,
    }


class VisionKnowledgeGraphBuilder:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def add_image(self, document: VisionDocument) -> str:
        image_node_id = self._graph_repository.create_node(
            EntityLabel.IMAGE,
            canonical_name=document.id,
            confidence=1.0,
            properties={
                "filename": document.filename,
                "source_type": document.source_type.value,
                "width": document.width,
                "height": document.height,
            },
        )
        if document.equipment_hint:
            entity = self._graph_repository.find_entity(document.equipment_hint)
            if entity is not None:
                self._graph_repository.create_relation(image_node_id, entity["id"], RelationType.DEPICTS, confidence=0.7)
        return image_node_id

    def add_ocr_regions(self, image_node_id: str, regions: list[OCRRegion]) -> list[str]:
        node_ids = []
        for region in regions:
            node_id = self._graph_repository.create_node(
                EntityLabel.OCR_REGION,
                canonical_name=str(uuid.uuid4()),
                confidence=region.confidence,
                properties=_node_properties(region.confidence, image_node_id, region.bounding_box, {"text": region.text}),
            )
            self._graph_repository.create_relation(node_id, image_node_id, RelationType.READ_FROM, confidence=region.confidence)
            node_ids.append(node_id)
        return node_ids

    def add_diagram_symbols(self, image_node_id: str, symbols: list[DiagramSymbol]) -> list[str]:
        node_ids = []
        for symbol in symbols:
            node_id = self._graph_repository.create_node(
                EntityLabel.DIAGRAM_SYMBOL,
                canonical_name=str(uuid.uuid4()),
                confidence=symbol.confidence,
                properties=_node_properties(symbol.confidence, image_node_id, symbol.bounding_box, {"symbol_type": symbol.symbol_type.value}),
            )
            self._graph_repository.create_relation(node_id, image_node_id, RelationType.DETECTED_IN, confidence=symbol.confidence)
            ontology_label = ontology_label_for_symbol(symbol.symbol_type)
            if ontology_label is not None:
                entity_node_id = self._graph_repository.create_node(ontology_label, canonical_name=f"{ontology_label.value}:{node_id[:8]}", confidence=symbol.confidence)
                self._graph_repository.create_relation(node_id, entity_node_id, RelationType.ASSOCIATED_WITH, confidence=symbol.confidence)
            node_ids.append(node_id)
        return node_ids

    def add_detections(self, image_node_id: str, detections: list[Detection]) -> list[str]:
        node_ids = []
        for detection in detections:
            node_id = self._graph_repository.create_node(
                EntityLabel.DETECTED_OBJECT,
                canonical_name=str(uuid.uuid4()),
                confidence=detection.confidence,
                properties=_node_properties(detection.confidence, image_node_id, detection.bounding_box, {"equipment_type": detection.equipment_type}),
            )
            self._graph_repository.create_relation(node_id, image_node_id, RelationType.DETECTED_IN, confidence=detection.confidence)
            if detection.equipment_type != "unknown":
                entity = self._graph_repository.find_entity(detection.equipment_type)
                if entity is not None:
                    self._graph_repository.create_relation(node_id, entity["id"], RelationType.ASSOCIATED_WITH, confidence=detection.confidence)
            node_ids.append(node_id)
        return node_ids

    def add_gauge_readings(self, image_node_id: str, readings: list[GaugeReading]) -> list[str]:
        node_ids = []
        for reading in readings:
            node_id = self._graph_repository.create_node(
                EntityLabel.GAUGE_READING,
                canonical_name=str(uuid.uuid4()),
                confidence=reading.confidence,
                properties=_node_properties(
                    reading.confidence, image_node_id, reading.bounding_box,
                    {"normalized_value": reading.normalized_value, "needle_angle_degrees": reading.needle_angle_degrees, "valid": reading.valid},
                ),
            )
            self._graph_repository.create_relation(node_id, image_node_id, RelationType.READ_FROM, confidence=reading.confidence)
            node_ids.append(node_id)
        return node_ids

    def add_defects(self, image_node_id: str, defects: list[Defect]) -> list[str]:
        node_ids = []
        for defect in defects:
            node_id = self._graph_repository.create_node(
                EntityLabel.DEFECT,
                canonical_name=str(uuid.uuid4()),
                confidence=defect.confidence,
                properties=_node_properties(defect.confidence, image_node_id, defect.bounding_box, {"defect_type": defect.defect_type.value, "severity": defect.severity}),
            )
            self._graph_repository.create_relation(node_id, image_node_id, RelationType.DETECTED_IN, confidence=defect.confidence)
            node_ids.append(node_id)
        return node_ids
