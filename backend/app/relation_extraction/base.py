"""Common interface every relation extractor must implement.

The rule-based implementation is deterministic (co-occurrence + keyword
cues); the interface is deliberately extractor-agnostic so a future
transformer-based (or LLM-assisted) relation extractor can be dropped in
without touching the knowledge graph builder that consumes ExtractedRelation.
"""

from typing import Protocol

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.relation_extraction.schemas import ExtractedRelation


class RelationExtractor(Protocol):
    def extract(self, entities: list[ExtractedEntity], chunk: Chunk) -> list[ExtractedRelation]: ...
