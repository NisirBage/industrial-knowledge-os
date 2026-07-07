"""Deterministic, sentence-scoped relation extraction.

For each sentence in a chunk, entities whose character spans fall inside that
sentence are paired up; the text between each pair is checked against a
keyword-cue table to decide the relation type. Two entities with no matching
cue but the only two entities in an otherwise relation-bearing sentence are
still linked as RELATED_TO — anything noisier than that (3+ entities, no cue)
is left unlinked rather than guessing, to keep the graph precise.

This is intentionally simple and auditable; the interface (base.py) is the
extension point for a future transformer/LLM-based relation extractor.
"""

import re
import uuid

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel, RelationType
from app.relation_extraction.schemas import ExtractedRelation

_STANDARD_LIKE_LABELS = {EntityLabel.STANDARD, EntityLabel.REGULATION}

# Ordered so more specific phrases are checked before generic ones.
_KEYWORD_CUES: list[tuple[re.Pattern, RelationType]] = [
    (re.compile(r"\bfailed due to\b|\bfailure due to\b", re.IGNORECASE), RelationType.FAILED_DUE_TO),
    (re.compile(r"\bcaused by\b", re.IGNORECASE), RelationType.CAUSES),
    (re.compile(r"\bcauses?\b|\bled to\b", re.IGNORECASE), RelationType.CAUSES),
    (re.compile(r"\bmitigated by\b|\bcorrected by\b|\bresolved by\b", re.IGNORECASE), RelationType.MITIGATED_BY),
    (re.compile(r"\binspected by\b", re.IGNORECASE), RelationType.INSPECTED_BY),
    (re.compile(r"\bmaintained by\b", re.IGNORECASE), RelationType.MAINTAINED_BY),
    (re.compile(r"\binstalled in\b", re.IGNORECASE), RelationType.INSTALLED_IN),
    (re.compile(r"\blocated in\b", re.IGNORECASE), RelationType.LOCATED_IN),
    (re.compile(r"\bpart of\b", re.IGNORECASE), RelationType.PART_OF),
    (re.compile(r"\bconnected to\b|\bconnects to\b|\bupstream of\b|\bdownstream of\b", re.IGNORECASE), RelationType.CONNECTED_TO),
    (re.compile(r"\brequires?\b", re.IGNORECASE), RelationType.REQUIRES),
    (re.compile(r"\bgenerated from\b", re.IGNORECASE), RelationType.GENERATED_FROM),
    (re.compile(r"\breferences?\b", re.IGNORECASE), RelationType.REFERENCES),
    (re.compile(r"\bper\b|\brequirements\b|\bunder\b", re.IGNORECASE), RelationType.USES_STANDARD),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for part in _SENTENCE_SPLIT.split(text):
        end = start + len(part)
        spans.append((start, end))
        start = end + 1  # account for the split whitespace
    return spans


class RuleBasedRelationExtractor:
    def extract(self, entities: list[ExtractedEntity], chunk: Chunk) -> list[ExtractedRelation]:
        if len(entities) < 2:
            return []

        relations: list[ExtractedRelation] = []
        for sentence_start, sentence_end in _sentence_spans(chunk.text):
            sentence_entities = sorted(
                (e for e in entities if sentence_start <= e.start_char < sentence_end),
                key=lambda e: e.start_char,
            )
            if len(sentence_entities) < 2:
                continue

            relations.extend(self._relations_for_sentence(sentence_entities, chunk))

        return relations

    def _relations_for_sentence(self, sentence_entities: list[ExtractedEntity], chunk: Chunk) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        any_cue_matched = False

        for i in range(len(sentence_entities) - 1):
            source = sentence_entities[i]
            target = sentence_entities[i + 1]
            between_text = chunk.text[source.end_char:target.start_char]

            relation_type = self._match_cue(between_text, target.label)
            if relation_type is not None:
                any_cue_matched = True
                relations.append(self._build(source, target, relation_type, chunk, confidence=0.7))

        if not any_cue_matched and len(sentence_entities) == 2:
            source, target = sentence_entities
            relations.append(self._build(source, target, RelationType.RELATED_TO, chunk, confidence=0.4))

        return relations

    @staticmethod
    def _match_cue(between_text: str, target_label: EntityLabel) -> RelationType | None:
        for pattern, relation_type in _KEYWORD_CUES:
            if relation_type == RelationType.USES_STANDARD and target_label not in _STANDARD_LIKE_LABELS:
                continue
            if pattern.search(between_text):
                return relation_type
        return None

    @staticmethod
    def _build(
        source: ExtractedEntity,
        target: ExtractedEntity,
        relation_type: RelationType,
        chunk: Chunk,
        confidence: float,
    ) -> ExtractedRelation:
        return ExtractedRelation(
            id=str(uuid.uuid4()),
            source_entity_id=source.id,
            target_entity_id=target.id,
            relation_type=relation_type,
            confidence=confidence,
            document_id=chunk.document_id,
            chunk_id=chunk.id,
            metadata={"extractor": "rule_based"},
        )
