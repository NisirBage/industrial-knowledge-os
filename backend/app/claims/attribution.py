"""Evidence Attribution Framework (O2 Component 4).

Builds a per-sentence attribution record for the Narrator's final answer —
reusing exactly what Faithfulness Scoring (Component 8) already computed
(which claim, if any, entails each answer sentence) rather than a second
NLI pass. Every field is a real lookup against already-built structures
(the verified Claim Graph, the Knowledge Graph via `GraphRepository`) —
never fabricated. "One-click inspection" is a frontend concern (the Chat
page's expandable per-sentence attribution panel); this module only
computes the underlying data each panel renders.
"""

from dataclasses import dataclass, field

from app.claims.models import Claim, ClaimGraph, ClaimRelation
from app.faithfulness.models import SentenceFaithfulness
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import GraphRepositoryProtocol, entity_id_for


@dataclass(frozen=True)
class SentenceAttribution:
    sentence: str
    supporting_claim_ids: list[str] = field(default_factory=list)
    supporting_evidence_ids: list[str] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)
    graph_edge_descriptions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning_path: str = ""


class EvidenceAttributionBuilder:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def build(self, sentence_scores: list[SentenceFaithfulness], claim_graph: ClaimGraph) -> list[SentenceAttribution]:
        return [self._build_one(score, claim_graph) for score in sentence_scores]

    def _build_one(self, score: SentenceFaithfulness, claim_graph: ClaimGraph) -> SentenceAttribution:
        if score.best_claim_id is None:
            return SentenceAttribution(sentence=score.sentence, reasoning_path="No verified claim entails this sentence.")

        claim = claim_graph.claim_by_id(score.best_claim_id)
        if claim is None:
            return SentenceAttribution(sentence=score.sentence, reasoning_path="Supporting claim reference is stale.")

        graph_node_ids = self._graph_node_ids(claim)
        edges = claim_graph.relations_from(claim.id) + claim_graph.relations_to(claim.id)
        edge_descriptions = [f"{r.source_claim_id} --{r.relation_type}--> {r.target_claim_id}" for r in edges]
        confidence = claim.propagated_confidence if claim.propagated_confidence is not None else claim.confidence

        return SentenceAttribution(
            sentence=score.sentence,
            supporting_claim_ids=[claim.id],
            supporting_evidence_ids=list(claim.source_evidence_ids),
            graph_node_ids=graph_node_ids,
            graph_edge_descriptions=edge_descriptions,
            confidence=round(confidence, 4),
            reasoning_path=self._reasoning_path(claim, edges),
        )

    def _graph_node_ids(self, claim: Claim) -> list[str]:
        try:
            label = EntityLabel(claim.object)
        except ValueError:
            return []
        try:
            if self._graph_repository.find_entity(claim.subject, label=label) is None:
                return []
        except Exception:
            return []
        return [entity_id_for(label, claim.subject)]

    @staticmethod
    def _reasoning_path(claim: Claim, edges: list[ClaimRelation]) -> str:
        chunk_part = ", ".join(claim.source_evidence_ids) or "no source chunk"
        path = f"Evidence[{chunk_part}] -> Claim[{claim.id}]({claim.verification_status.value})"
        if edges:
            path += f" -> {len(edges)} related claim(s)"
        return path + " -> Answer sentence"
