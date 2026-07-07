"""Entity extraction pipeline: extract (primary + supplement) -> deduplicate
-> normalize -> optional LLM validation -> structured entities.

This is the single entry point ingestion calls; it never talks to GLiNER,
regex, or the LLM directly, so any of those can be swapped without touching
ingestion_service.py.
"""

import dataclasses

from app.documents.chunking import Chunk
from app.entity_extraction.base import EntityExtractor
from app.entity_extraction.llm_validator import LlmEntityValidator
from app.entity_extraction.normalizer import normalize_canonical_name
from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import EntityLabel


class EntityExtractionService:
    def __init__(
        self,
        primary: EntityExtractor,
        supplement: EntityExtractor | None = None,
        validator: LlmEntityValidator | None = None,
    ):
        self._primary = primary
        self._supplement = supplement
        self._validator = validator

    def extract(self, chunk: Chunk) -> list[ExtractedEntity]:
        raw = list(self._primary.extract(chunk))
        if self._supplement is not None:
            raw.extend(self._supplement.extract(chunk))

        deduplicated = self._deduplicate(raw)

        if self._validator is not None:
            deduplicated = self._validator.validate(deduplicated, chunk)

        return deduplicated

    @staticmethod
    def _deduplicate(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
        merged: dict[tuple[EntityLabel, str], ExtractedEntity] = {}

        for entity in entities:
            normalized_name = normalize_canonical_name(entity.label, entity.canonical_name)
            key = (entity.label, normalized_name)

            if key not in merged:
                merged[key] = dataclasses.replace(entity, canonical_name=normalized_name)
                continue

            existing = merged[key]
            merged[key] = dataclasses.replace(
                existing,
                aliases=existing.aliases | entity.aliases | {entity.canonical_name},
                confidence=max(existing.confidence, entity.confidence),
            )

        return list(merged.values())
