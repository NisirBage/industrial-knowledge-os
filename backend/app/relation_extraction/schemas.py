from dataclasses import dataclass, field

from app.ontology.labels import RelationType


@dataclass(frozen=True)
class ExtractedRelation:
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    confidence: float
    document_id: str
    chunk_id: str
    metadata: dict = field(default_factory=dict)
