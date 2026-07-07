"""Common interface every entity extractor must implement.

This is the seam that lets GLiNER, regex, spaCy, or a custom transformer be
swapped or combined without ingestion ever knowing which one ran.
"""

from typing import Protocol

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity


class EntityExtractor(Protocol):
    def extract(self, chunk: Chunk) -> list[ExtractedEntity]: ...
