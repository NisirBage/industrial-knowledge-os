"""Claim Extraction Engine (O2 Component 2).

Converts every piece of evidence into atomic claims by reusing the exact
same `EntityExtractionService` / `RelationExtractor` collaborators the
ingestion pipeline already uses — no second, parallel NLP stack. Each
extracted entity becomes a FACT claim ("Equipment = Pump P-101"); each
extracted relation between two entities becomes a `ClaimRelation` directly
between the two entity claims, reusing the same `RelationType` value the
Knowledge Graph itself uses (e.g. CAUSES) — so "Pump P101 failed because of
cavitation" yields exactly the worked example from the spec:

    Claim A: Equipment = Pump P-101      (FACT)
    Claim B: Failure Mode = Cavitation   (FACT)
    ClaimRelation(A -> B, CAUSES)

RISK and HYPOTHESIS claims are additionally mirrored 1:1 from the Industrial
Reasoning Graph's own RISK/HYPOTHESIS nodes (already computed by the prior
`reasoning` stage from causal relation triples and detected contradictions —
see app/reasoning/reasoning_graph_builder.py) rather than re-derived here,
linked back to whichever FACT claims share a source chunk id via a `SUPPORTS`
ClaimRelation (reusing the ontology's own SUPPORTS relation type).

Two of the spec's four listed extraction "supports" are honestly out of
scope today: real dependency parsing (the relation extractor is sentence-
scoped keyword-cue matching, not a dependency parse) and a future LLM
extractor — both documented extension points, not implemented, consistent
with this codebase's anti-fabrication precedent (see Roadmap).

PREDICTION and RECOMMENDATION claim types are intentionally not populated
here: no Prediction Engine call happens inside the chat pipeline today (see
docs/architecture/reasoning.md), and RECOMMENDATION claims are added later,
once real Recommendation objects exist, by Recommendation Verification
(app/recommendation/verification.py) — not fabricated in advance here.
"""

from app.claims.models import Claim, ClaimGraph, ClaimRelation, ClaimSpan, ClaimType
from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.entity_extraction.service import EntityExtractionService
from app.reasoning.graph_models import ReasoningGraph, ReasoningNodeType
from app.relation_extraction.base import RelationExtractor
from app.retrieval.evidence import Evidence

_REASONING_TO_CLAIM_TYPE = {
    ReasoningNodeType.RISK: ClaimType.RISK,
    ReasoningNodeType.HYPOTHESIS: ClaimType.HYPOTHESIS,
}


class ClaimExtractionEngine:
    def __init__(self, entity_extraction_service: EntityExtractionService, relation_extractor: RelationExtractor):
        self._entity_extraction_service = entity_extraction_service
        self._relation_extractor = relation_extractor

    def extract(self, evidence_items: list[Evidence], reasoning_graph: ReasoningGraph | None = None) -> ClaimGraph:
        graph = ClaimGraph(claims=[], relations=[])
        fact_claim_ids_by_chunk: dict[str, list[str]] = {}

        for evidence in evidence_items:
            chunk = self._chunk_from_evidence(evidence)
            entities = self._entity_extraction_service.extract(chunk)
            entity_claim_ids = self._add_entity_claims(graph, entities, evidence)
            fact_claim_ids_by_chunk.setdefault(evidence.chunk_id, []).extend(entity_claim_ids.values())

            if len(entities) >= 2:
                relations = self._relation_extractor.extract(entities, chunk)
                self._add_relation_edges(graph, relations, entity_claim_ids)

        if reasoning_graph is not None:
            self._add_reasoning_graph_claims(graph, reasoning_graph, fact_claim_ids_by_chunk)

        return graph

    @staticmethod
    def _chunk_from_evidence(evidence: Evidence) -> Chunk:
        return Chunk(
            id=evidence.chunk_id,
            document_id=evidence.document_id,
            text=evidence.text,
            page_number=evidence.page_number,
            chunk_index=0,
        )

    @staticmethod
    def _add_entity_claims(graph: ClaimGraph, entities: list[ExtractedEntity], evidence: Evidence) -> dict[str, str]:
        entity_claim_ids: dict[str, str] = {}
        for entity in entities:
            claim_id = f"claim:entity:{evidence.chunk_id}:{entity.id}"
            label_text = entity.label.value.replace("_", " ").title()
            graph.add_claim(
                Claim(
                    id=claim_id,
                    claim_type=ClaimType.FACT,
                    text=f"{label_text} = {entity.canonical_name}",
                    subject=entity.canonical_name,
                    predicate="is_a",
                    object=entity.label.value,
                    confidence=entity.confidence,
                    source_evidence_ids=[evidence.chunk_id],
                    source_spans=[
                        ClaimSpan(
                            chunk_id=evidence.chunk_id,
                            document_id=evidence.document_id,
                            start_char=entity.start_char,
                            end_char=entity.end_char,
                        )
                    ],
                )
            )
            entity_claim_ids[entity.id] = claim_id
        return entity_claim_ids

    @staticmethod
    def _add_relation_edges(graph: ClaimGraph, relations, entity_claim_ids: dict[str, str]) -> None:
        for relation in relations:
            source_claim_id = entity_claim_ids.get(relation.source_entity_id)
            target_claim_id = entity_claim_ids.get(relation.target_entity_id)
            if source_claim_id is None or target_claim_id is None:
                continue
            graph.add_relation(
                ClaimRelation(
                    id=f"claim-relation:{relation.id}",
                    source_claim_id=source_claim_id,
                    target_claim_id=target_claim_id,
                    relation_type=relation.relation_type.value,
                    confidence=relation.confidence,
                )
            )

    @staticmethod
    def _add_reasoning_graph_claims(
        graph: ClaimGraph,
        reasoning_graph: ReasoningGraph,
        fact_claim_ids_by_chunk: dict[str, list[str]],
    ) -> None:
        for node_type, claim_type in _REASONING_TO_CLAIM_TYPE.items():
            for node in reasoning_graph.nodes_of_type(node_type):
                claim_id = f"claim:{node_type.value}:{node.id}"
                graph.add_claim(
                    Claim(
                        id=claim_id,
                        claim_type=claim_type,
                        text=node.description,
                        subject=node.description,
                        predicate=node_type.value,
                        confidence=node.confidence,
                        source_evidence_ids=list(node.supporting_evidence_ids),
                    )
                )
                for chunk_id in node.supporting_evidence_ids:
                    for fact_claim_id in fact_claim_ids_by_chunk.get(chunk_id, []):
                        graph.add_relation(
                            ClaimRelation(
                                id=f"claim-relation:supports:{fact_claim_id}:{claim_id}",
                                source_claim_id=fact_claim_id,
                                target_claim_id=claim_id,
                                relation_type="SUPPORTS",
                                confidence=node.confidence,
                            )
                        )
