from dataclasses import dataclass, field

from app.ontology.labels import EntityLabel


@dataclass(frozen=True)
class ExtractedEntity:
    id: str
    label: EntityLabel
    canonical_name: str
    aliases: frozenset[str]
    confidence: float
    document_id: str
    chunk_id: str
    page_number: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
