"""Knowledge Graph Builder: the only thing between ingestion and the graph
repository. Owns node creation, deduplication (delegated to the repository's
deterministic entity ids), relationship creation, and the document-mentions
linking. Ingestion never calls GraphRepository directly — it calls
KnowledgeGraphBuilder.build(), so no Neo4j/Cypher concerns leak into the
ingestion pipeline.

"Versioning" here means every relationship accumulates observation metadata
(confidence high-water mark, contributing chunk ids, observation_count,
first/last seen) across repeated ingestion runs rather than being overwritten
— see GraphRepository.create_relation.
"""

from dataclasses import dataclass

from app.entity_extraction.schemas import ExtractedEntity
from app.ontology.labels import RelationType
from app.relation_extraction.schemas import ExtractedRelation
from app.repositories.graph_repository import GraphRepositoryProtocol


@dataclass(frozen=True)
class KnowledgeGraphBuildResult:
    document_node_id: str
    entity_node_ids: dict[str, str]  # ExtractedEntity.id -> graph node id
    relations_created: int


class KnowledgeGraphBuilder:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def build(
        self,
        document_id: str,
        filename: str,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> KnowledgeGraphBuildResult:
        document_node_id = self._graph_repository.upsert_document(document_id, filename)
        entity_node_ids = self._link_entities(document_node_id, entities)
        relations_created = self._link_relations(relations, entity_node_ids)

        return KnowledgeGraphBuildResult(
            document_node_id=document_node_id,
            entity_node_ids=entity_node_ids,
            relations_created=relations_created,
        )

    def _link_entities(self, document_node_id: str, entities: list[ExtractedEntity]) -> dict[str, str]:
        entity_node_ids: dict[str, str] = {}
        for entity in entities:
            node_id = self._graph_repository.create_node(
                label=entity.label,
                canonical_name=entity.canonical_name,
                aliases=entity.aliases,
                confidence=entity.confidence,
            )
            self._graph_repository.create_relation(
                document_node_id,
                node_id,
                RelationType.MENTIONS,
                confidence=1.0,
                chunk_id=entity.chunk_id,
            )
            entity_node_ids[entity.id] = node_id
        return entity_node_ids

    def _link_relations(self, relations: list[ExtractedRelation], entity_node_ids: dict[str, str]) -> int:
        created = 0
        for relation in relations:
            source_node_id = entity_node_ids.get(relation.source_entity_id)
            target_node_id = entity_node_ids.get(relation.target_entity_id)
            if source_node_id is None or target_node_id is None:
                continue
            self._graph_repository.create_relation(
                source_node_id,
                target_node_id,
                relation.relation_type,
                confidence=relation.confidence,
                chunk_id=relation.chunk_id,
            )
            created += 1
        return created
