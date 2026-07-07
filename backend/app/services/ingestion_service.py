"""Orchestrates the ingestion pipeline:
load -> chunk -> embed -> vector store -> entity extraction -> relation
extraction -> knowledge graph builder.

Never talks to Neo4j/Cypher directly — all graph writes go through
KnowledgeGraphBuilder, and all entity/relation extraction goes through the
EntityExtractionService/RelationExtractor interfaces, so any of those layers
can be swapped without this module changing.
"""

from pathlib import Path

from app.documents.chunking import chunk_pages
from app.documents.loaders import load_document
from app.embeddings.embedder import Embedder
from app.entity_extraction.service import EntityExtractionService
from app.graph.knowledge_graph_builder import KnowledgeGraphBuilder
from app.relation_extraction.base import RelationExtractor
from app.repositories.metadata_repository import MetadataRepository
from app.repositories.vector_repository import VectorRepository
from app.retrieval.bm25_index import Bm25Index
from app.schemas.document import DocumentUploadResponse


class IngestionService:
    def __init__(
        self,
        embedder: Embedder,
        vector_repository: VectorRepository,
        metadata_repository: MetadataRepository,
        bm25_index: Bm25Index,
        entity_extraction_service: EntityExtractionService,
        relation_extractor: RelationExtractor,
        knowledge_graph_builder: KnowledgeGraphBuilder,
    ):
        self._embedder = embedder
        self._vector_repository = vector_repository
        self._metadata_repository = metadata_repository
        self._bm25_index = bm25_index
        self._entity_extraction_service = entity_extraction_service
        self._relation_extractor = relation_extractor
        self._knowledge_graph_builder = knowledge_graph_builder

    async def ingest(self, file_path: Path, original_filename: str, content_type: str) -> DocumentUploadResponse:
        document_id = await self._metadata_repository.create_document(original_filename, content_type)

        document = load_document(file_path)
        chunks = chunk_pages(document, document_id=document_id)

        if chunks:
            vectors = self._embedder.embed_texts([c.text for c in chunks])
            self._vector_repository.ensure_collection(vector_size=self._embedder.dimension)
            self._vector_repository.upsert_chunks(chunks, vectors)
            self._bm25_index.add_chunks(chunks)

        entities = [entity for chunk in chunks for entity in self._entity_extraction_service.extract(chunk)]
        relations = [
            relation
            for chunk in chunks
            for relation in self._relation_extractor.extract(
                [e for e in entities if e.chunk_id == chunk.id], chunk
            )
        ]

        build_result = self._knowledge_graph_builder.build(document_id, original_filename, entities, relations)

        await self._metadata_repository.update_counts(document_id, len(chunks), len(entities))

        return DocumentUploadResponse(
            document_id=document_id,
            filename=original_filename,
            num_chunks=len(chunks),
            num_entities=len(entities),
            num_relations=build_result.relations_created,
        )
