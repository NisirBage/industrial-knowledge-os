"""Multimodal Retrieval (M7 Component 8).

Merges the existing text HybridRetriever's evidence with visual evidence
(OCR text, diagram symbols, gauge readings, defects, detections) associated
with the query's entities — into ONE evidence list, never a separate
"image answer." Visual evidence is emitted as ordinary `Evidence` instances
(the same dataclass text retrieval already produces), so
EvidenceFusionEngine / ContradictionDetector / ReasoningGraphBuilder consume
it identically to text evidence with zero changes — "never bypass Evidence
Fusion" from the M7 spec, satisfied structurally rather than by convention.
"""

import logging

from app.documents.chunking import Chunk
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.retrieval.evidence import Evidence
from app.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

_VISION_NODE_TYPES = {
    EntityLabel.IMAGE.value,
    EntityLabel.OCR_REGION.value,
    EntityLabel.DETECTED_OBJECT.value,
    EntityLabel.DEFECT.value,
    EntityLabel.DIAGRAM_SYMBOL.value,
    EntityLabel.GAUGE_READING.value,
}


def _describe_node(node_type: str, properties: dict) -> str:
    if node_type == EntityLabel.OCR_REGION.value:
        return f"OCR read the text '{properties.get('text', '')}' in an image."
    if node_type == EntityLabel.DEFECT.value:
        return f"Visual inspection detected a {properties.get('severity', 'unknown')}-severity {properties.get('defect_type', 'defect')} in an image."
    if node_type == EntityLabel.GAUGE_READING.value:
        value = properties.get("normalized_value")
        if value is None:
            return "A gauge reading was attempted in an image but could not be validated."
        return f"A gauge reading of {value * 100:.1f}% of full scale was observed in an image."
    if node_type == EntityLabel.DETECTED_OBJECT.value:
        return f"An object of type '{properties.get('equipment_type', 'unknown')}' was visually detected in an image."
    if node_type == EntityLabel.DIAGRAM_SYMBOL.value:
        return f"A '{properties.get('symbol_type', 'unknown')}' symbol was detected in an engineering diagram."
    if node_type == EntityLabel.IMAGE.value:
        return f"An image ({properties.get('source_type', 'unknown')}) depicting this equipment is on file."
    return "Visual evidence was found for this equipment."


class MultimodalRetriever:
    def __init__(
        self,
        text_retriever: HybridRetriever,
        graph_repository: GraphRepositoryProtocol,
        entity_extraction_service: EntityExtractionService,
    ):
        self._text_retriever = text_retriever
        self._graph_repository = graph_repository
        self._entity_extraction_service = entity_extraction_service

    def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
        # Same "log and degrade, don't crash the whole response" discipline
        # TaskGraphExecutor already applies to every chat retrieval node —
        # a fresh deployment with no documents ingested yet is a legitimate
        # state, not a bug, and a multimodal query in that state should still
        # return whatever visual evidence exists rather than a 500.
        try:
            text_evidence = self._text_retriever.retrieve(query, top_k=top_k)
        except Exception:
            logger.exception("Text retrieval failed for multimodal query %r — returning visual evidence only", query)
            text_evidence = []
        visual_evidence = self._visual_evidence_for_query(query)
        return text_evidence + visual_evidence

    def _visual_evidence_for_query(self, query: str) -> list[Evidence]:
        query_chunk = Chunk(id="query", document_id="query", text=query, page_number=0, chunk_index=0)
        entities = self._entity_extraction_service.extract(query_chunk)

        evidence: list[Evidence] = []
        seen_node_ids: set[str] = set()
        for entity in entities:
            node = self._graph_repository.find_entity(entity.canonical_name, entity.label)
            if node is None:
                continue
            evidence += self._visual_evidence_for_entity(node["id"], seen_node_ids)
        return evidence

    def _visual_evidence_for_entity(self, entity_id: str, seen_node_ids: set[str]) -> list[Evidence]:
        found: list[Evidence] = []

        directly_associated = self._graph_repository.find_neighbors(entity_id, relation_type=RelationType.ASSOCIATED_WITH, depth=1)
        for node in directly_associated:
            if node["type"] in _VISION_NODE_TYPES and node["id"] not in seen_node_ids:
                seen_node_ids.add(node["id"])
                found.append(self._node_to_evidence(node))

        depicting_images = self._graph_repository.find_neighbors(entity_id, relation_type=RelationType.DEPICTS, depth=1)
        for image_node in depicting_images:
            if image_node["type"] != EntityLabel.IMAGE.value:
                continue
            for node in self._graph_repository.find_neighbors(image_node["id"], depth=1):
                if node["type"] in _VISION_NODE_TYPES and node["id"] not in seen_node_ids:
                    seen_node_ids.add(node["id"])
                    found.append(self._node_to_evidence(node))
        return found

    def _node_to_evidence(self, node: dict) -> Evidence:
        properties = self._graph_repository.get_node_properties(node["id"])
        confidence = properties.get("confidence", 0.5)
        return Evidence(
            document_id=properties.get("image_node_id", ""),
            page_number=0,
            chunk_id=node["id"],
            text=_describe_node(node["type"], properties),
            entity_ids=[node["id"]],
            graph_score=confidence,
            confidence=confidence,
            source=f"vision:{node['type']}",
        )
