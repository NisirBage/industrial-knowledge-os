"""GLiNER-backed zero-shot entity extractor — the primary extraction strategy.

GLiNER accepts arbitrary label strings at inference time (no fine-tuning per
label), so the ontology's labels can be handed to it directly as a prompt-like
label set. Construction downloads a pretrained model; if that fails (no
network, no local cache) the factory falls back to regex-only, consistent
with the offline-resilience pattern used for embeddings and the LLM provider.
"""

import uuid

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel

_GLINER_LABEL_TO_ONTOLOGY: dict[str, EntityLabel] = {
    "pump": EntityLabel.PUMP,
    "motor": EntityLabel.MOTOR,
    "compressor": EntityLabel.COMPRESSOR,
    "valve": EntityLabel.VALVE,
    "pipe": EntityLabel.PIPE,
    "tank": EntityLabel.TANK,
    "sensor": EntityLabel.SENSOR,
    "instrument": EntityLabel.INSTRUMENT,
    "maintenance record": EntityLabel.MAINTENANCE_RECORD,
    "inspection": EntityLabel.INSPECTION,
    "incident": EntityLabel.INCIDENT,
    "failure mode": EntityLabel.FAILURE_MODE,
    "corrective action": EntityLabel.CORRECTIVE_ACTION,
    "preventive action": EntityLabel.PREVENTIVE_ACTION,
    "permit": EntityLabel.PERMIT,
    "procedure": EntityLabel.PROCEDURE,
    "regulation": EntityLabel.REGULATION,
    "standard": EntityLabel.STANDARD,
    "manufacturer": EntityLabel.MANUFACTURER,
    "location": EntityLabel.LOCATION,
    "department": EntityLabel.DEPARTMENT,
    "person": EntityLabel.PERSONNEL,
}


class GlinerEntityExtractor:
    def __init__(self, model_name: str, threshold: float = 0.4):
        from gliner import GLiNER

        self._model = GLiNER.from_pretrained(model_name)
        self._labels = list(_GLINER_LABEL_TO_ONTOLOGY.keys())
        self._threshold = threshold

    def extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        predictions = self._model.predict_entities(chunk.text, self._labels, threshold=self._threshold)
        entities: list[ExtractedEntity] = []
        for prediction in predictions:
            label = _GLINER_LABEL_TO_ONTOLOGY.get(prediction["label"])
            if label is None:
                continue
            text = prediction["text"].strip()
            if not text:
                continue
            entities.append(
                ExtractedEntity(
                    id=str(uuid.uuid4()),
                    label=label,
                    canonical_name=text,
                    aliases=frozenset({text}),
                    confidence=float(prediction.get("score", 0.5)),
                    document_id=chunk.document_id,
                    chunk_id=chunk.id,
                    page_number=chunk.page_number,
                    start_char=prediction["start"],
                    end_char=prediction["end"],
                    metadata={"extractor": "gliner"},
                )
            )
        return entities
