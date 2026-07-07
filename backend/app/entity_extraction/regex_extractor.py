"""Deterministic regex extractor.

This was the primary (only) extraction strategy in the MVP. It is now a
fallback/supplement to GLiNER: regex is precise for structured codes (tag
numbers, standard/regulation names) that a zero-shot NER model can miss or
mis-segment, so it always runs alongside GLiNER rather than being replaced
by it (see entity_extraction/factory.py).
"""

import re
import uuid

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel

_EQUIPMENT_PATTERN = re.compile(r"\b([A-Z]{1,4})-\d{2,5}[A-Z]?\b")
_STANDARD_PATTERN = re.compile(
    r"\b(ISO\s?\d{3,5}(?::\d{4})?|BICSI|TIA-942)\b", re.IGNORECASE
)
_REGULATION_PATTERN = re.compile(
    r"\b(OSHA|OISD-?\d{0,4}|DGMS|DGFASLI|Factory Act|PESO)\b", re.IGNORECASE
)

# Common ISA-5.1-style tag prefixes -> equipment type. Anything not in this
# table maps to the generic INDUSTRIAL_ASSET label rather than guessing.
_TAG_PREFIX_TO_LABEL: dict[str, EntityLabel] = {
    "P": EntityLabel.PUMP,
    "M": EntityLabel.MOTOR,
    "C": EntityLabel.COMPRESSOR,
    "V": EntityLabel.VALVE,
    "T": EntityLabel.TANK,
    "S": EntityLabel.SENSOR,
}


def _label_for_tag(prefix: str) -> EntityLabel:
    return _TAG_PREFIX_TO_LABEL.get(prefix.upper(), EntityLabel.INDUSTRIAL_ASSET)


class RegexEntityExtractor:
    def extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        entities: list[ExtractedEntity] = []
        seen_spans: set[tuple[int, int]] = set()

        # Regulation/standard patterns are checked first and claim their
        # spans: they're specific named acronyms, whereas the equipment-tag
        # pattern is a generic heuristic ([A-Z]{1,4}-\d+) that would otherwise
        # also match codes like "OISD-105" and mislabel them as equipment.
        for match in _REGULATION_PATTERN.finditer(chunk.text):
            entities.append(self._build(chunk, match, EntityLabel.REGULATION))
            seen_spans.add(match.span())

        for match in _STANDARD_PATTERN.finditer(chunk.text):
            if match.span() in seen_spans:
                continue
            entities.append(self._build(chunk, match, EntityLabel.STANDARD))
            seen_spans.add(match.span())

        for match in _EQUIPMENT_PATTERN.finditer(chunk.text):
            if match.span() in seen_spans:
                continue
            entities.append(self._build(chunk, match, _label_for_tag(match.group(1))))

        return entities

    @staticmethod
    def _build(chunk: Chunk, match: re.Match, label: EntityLabel) -> ExtractedEntity:
        text = match.group().strip()
        return ExtractedEntity(
            id=str(uuid.uuid4()),
            label=label,
            canonical_name=text,
            aliases=frozenset({text}),
            confidence=0.85,
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            page_number=chunk.page_number,
            start_char=match.start(),
            end_char=match.end(),
            metadata={"extractor": "regex"},
        )
