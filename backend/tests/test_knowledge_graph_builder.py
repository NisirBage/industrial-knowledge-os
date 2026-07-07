import uuid

from app.entity_extraction.schemas import ExtractedEntity
from app.graph.knowledge_graph_builder import KnowledgeGraphBuilder
from app.ontology.labels import EntityLabel, RelationType
from app.relation_extraction.schemas import ExtractedRelation
from app.repositories.graph_repository import InMemoryGraphRepository


def _entity(label: EntityLabel, canonical_name: str, chunk_id: str = "chunk-1") -> ExtractedEntity:
    return ExtractedEntity(
        id=str(uuid.uuid4()),
        label=label,
        canonical_name=canonical_name,
        aliases=frozenset({canonical_name}),
        confidence=0.8,
        document_id="doc-1",
        chunk_id=chunk_id,
        page_number=1,
        start_char=0,
        end_char=len(canonical_name),
    )


def test_build_links_document_to_entities_and_creates_relations():
    repo = InMemoryGraphRepository()
    builder = KnowledgeGraphBuilder(repo)

    pump = _entity(EntityLabel.PUMP, "P-101")
    valve = _entity(EntityLabel.VALVE, "V-2201A")
    relation = ExtractedRelation(
        id=str(uuid.uuid4()),
        source_entity_id=pump.id,
        target_entity_id=valve.id,
        relation_type=RelationType.CONNECTED_TO,
        confidence=0.7,
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    result = builder.build("doc-1", "manual.pdf", [pump, valve], [relation])

    assert result.relations_created == 1
    assert set(result.entity_node_ids.keys()) == {pump.id, valve.id}

    mentioned = repo.entities_for_document("doc-1")
    assert {m["canonical_name"] for m in mentioned} == {"P-101", "V-2201A"}

    neighbors = repo.find_neighbors(result.entity_node_ids[pump.id], relation_type=RelationType.CONNECTED_TO)
    assert {n["canonical_name"] for n in neighbors} == {"V-2201A"}


def test_build_skips_relations_referencing_unknown_entities():
    repo = InMemoryGraphRepository()
    builder = KnowledgeGraphBuilder(repo)

    pump = _entity(EntityLabel.PUMP, "P-101")
    dangling_relation = ExtractedRelation(
        id=str(uuid.uuid4()),
        source_entity_id=pump.id,
        target_entity_id="not-a-real-entity-id",
        relation_type=RelationType.CONNECTED_TO,
        confidence=0.7,
        document_id="doc-1",
        chunk_id="chunk-1",
    )

    result = builder.build("doc-1", "manual.pdf", [pump], [dangling_relation])

    assert result.relations_created == 0
