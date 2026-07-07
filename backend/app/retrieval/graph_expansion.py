"""Graph expansion: the step that lets retrieval reason over structured
knowledge before vector/BM25 search finishes, per the graph-aware retrieval
architecture (Question -> Intent -> Metadata Filter -> Graph Expansion ->
Vector -> BM25 -> Merge -> Rerank).

Given the query text, this extracts entities from it (using the same
EntityExtractionService ingestion uses, for consistent recall), looks each up
in the knowledge graph, and returns:
  - documents the graph confirms mention those entities (used to boost
    vector/BM25 candidates from the same documents), and
  - synthesized "graph evidence" — human-readable relation triples (e.g.
    "P-101 CONNECTED_TO V-2201A") that participate in reranking directly,
    not just as a score boost, so relational knowledge the text chunks alone
    wouldn't surface can still reach the answer.
"""

from dataclasses import dataclass

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import RelationType
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.retrieval.evidence import Evidence

_GRAPH_EVIDENCE_CONFIDENCE = 0.55
# MENTIONS is document-to-entity bookkeeping (every entity mentions its
# source document trivially) — not domain knowledge, so it's excluded from
# synthesized graph evidence even though it still powers boosted_document_ids.
_EXCLUDED_FROM_EVIDENCE = {RelationType.MENTIONS.value}


@dataclass(frozen=True)
class GraphExpansionResult:
    matched_entity_ids: list[str]
    boosted_document_ids: set[str]
    graph_evidence: list[Evidence]


class GraphExpansionStage:
    def __init__(self, entity_extraction_service: EntityExtractionService, graph_repository: GraphRepositoryProtocol):
        self._entity_extraction_service = entity_extraction_service
        self._graph_repository = graph_repository

    def expand(self, query: str, max_neighbors_per_entity: int = 5) -> GraphExpansionResult:
        query_chunk = Chunk(id="query", document_id="query", text=query, page_number=0, chunk_index=0)
        query_entities = self._entity_extraction_service.extract(query_chunk)
        return self.expand_from_entities(query_entities, max_neighbors_per_entity)

    def expand_from_entities(
        self, query_entities: list[ExtractedEntity], max_neighbors_per_entity: int = 5
    ) -> GraphExpansionResult:
        """Same as `expand`, but skips re-extracting entities from the query
        text — used when a caller (e.g. KnowledgeGraphAgent) already has the
        query's entities from the Planner, so extraction runs exactly once
        per request instead of once per agent that needs it."""
        matched_entity_ids: list[str] = []
        boosted_document_ids: set[str] = set()
        graph_evidence: list[Evidence] = []

        for entity in query_entities:
            node = self._graph_repository.find_entity(entity.canonical_name, entity.label)
            if node is None:
                continue
            matched_entity_ids.append(node["id"])

            for document in self._graph_repository.documents_mentioning(entity.canonical_name):
                boosted_document_ids.add(document["document_id"])

            neighbors = self._graph_repository.find_neighbors(node["id"], depth=1)[:max_neighbors_per_entity]
            for neighbor in neighbors:
                relation_type = neighbor.get("relation_type")
                if relation_type is None or relation_type in _EXCLUDED_FROM_EVIDENCE:
                    continue
                graph_evidence.append(self._build_evidence(node, neighbor))

        return GraphExpansionResult(matched_entity_ids, boosted_document_ids, graph_evidence)

    @staticmethod
    def _build_evidence(source_node: dict, neighbor_node: dict) -> Evidence:
        text = f"{source_node['canonical_name']} {neighbor_node['relation_type']} {neighbor_node['canonical_name']}"
        return Evidence(
            document_id="",
            page_number=0,
            chunk_id=f"graph:{source_node['id']}:{neighbor_node['id']}",
            text=text,
            entity_ids=[source_node["id"], neighbor_node["id"]],
            graph_score=_GRAPH_EVIDENCE_CONFIDENCE,
            source="graph",
            confidence=_GRAPH_EVIDENCE_CONFIDENCE,
            agent="knowledge_graph",
        )
