from app.claims.attribution import EvidenceAttributionBuilder
from app.claims.models import Claim, ClaimGraph, ClaimRelation, ClaimType, ClaimVerificationStatus
from app.faithfulness.models import SentenceFaithfulness
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import InMemoryGraphRepository


def _claim(claim_id: str, subject: str = "P-101", object_: str = EntityLabel.PUMP.value, chunk_id: str = "c1") -> Claim:
    return Claim(
        id=claim_id,
        claim_type=ClaimType.FACT,
        text=f"Equipment = {subject}",
        subject=subject,
        predicate="is_a",
        object=object_,
        confidence=0.6,
        source_evidence_ids=[chunk_id],
        verification_status=ClaimVerificationStatus.SUPPORTED,
    )


def test_sentence_with_no_entailing_claim_has_empty_attribution():
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())
    scores = [SentenceFaithfulness(sentence="Unrelated text.", entailed=False, best_claim_id=None, score=0.0)]

    attributions = builder.build(scores, ClaimGraph(claims=[], relations=[]))

    assert attributions[0].supporting_claim_ids == []
    assert "No verified claim" in attributions[0].reasoning_path


def test_sentence_attribution_includes_claim_and_evidence_ids():
    claim = _claim("c1")
    claim_graph = ClaimGraph(claims=[claim], relations=[])
    scores = [SentenceFaithfulness(sentence="Pump P-101 is running.", entailed=True, best_claim_id="c1", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)
    attribution = attributions[0]

    assert attribution.supporting_claim_ids == ["c1"]
    assert attribution.supporting_evidence_ids == ["c1"]
    assert attribution.confidence == 0.6
    assert "Claim[c1]" in attribution.reasoning_path


def test_graph_node_ids_populated_when_entity_exists_in_knowledge_graph():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    claim = _claim("c1")
    claim_graph = ClaimGraph(claims=[claim], relations=[])
    scores = [SentenceFaithfulness(sentence="Pump P-101 is running.", entailed=True, best_claim_id="c1", score=0.9)]
    builder = EvidenceAttributionBuilder(repo)

    attributions = builder.build(scores, claim_graph)

    assert len(attributions[0].graph_node_ids) == 1


def test_graph_node_ids_empty_when_entity_not_in_knowledge_graph():
    claim = _claim("c1")
    claim_graph = ClaimGraph(claims=[claim], relations=[])
    scores = [SentenceFaithfulness(sentence="Pump P-101 is running.", entailed=True, best_claim_id="c1", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)

    assert attributions[0].graph_node_ids == []


def test_graph_edge_descriptions_include_related_claim_relations():
    source = _claim("c1", subject="P-101")
    target = _claim("c2", subject="V-2201A", object_=EntityLabel.VALVE.value, chunk_id="c1")
    claim_graph = ClaimGraph(
        claims=[source, target],
        relations=[ClaimRelation(id="r1", source_claim_id="c1", target_claim_id="c2", relation_type="FAILED_DUE_TO")],
    )
    scores = [SentenceFaithfulness(sentence="Pump P-101 failed.", entailed=True, best_claim_id="c1", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)

    assert len(attributions[0].graph_edge_descriptions) == 1
    assert "FAILED_DUE_TO" in attributions[0].graph_edge_descriptions[0]
    assert "2 related claim" not in attributions[0].reasoning_path  # only 1 relation, singular count
    assert "1 related claim(s)" in attributions[0].reasoning_path


def test_stale_claim_reference_is_handled_without_crashing():
    scores = [SentenceFaithfulness(sentence="Text.", entailed=True, best_claim_id="missing", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, ClaimGraph(claims=[], relations=[]))

    assert attributions[0].supporting_claim_ids == []
    assert "stale" in attributions[0].reasoning_path


def test_incoming_relations_are_also_included_in_edge_descriptions():
    source = _claim("c1", subject="P-101")
    target = _claim("c2", subject="V-2201A", object_=EntityLabel.VALVE.value, chunk_id="c1")
    claim_graph = ClaimGraph(
        claims=[source, target],
        relations=[ClaimRelation(id="r1", source_claim_id="c1", target_claim_id="c2", relation_type="FAILED_DUE_TO")],
    )
    # attributing the TARGET claim's sentence should also show the incoming edge.
    scores = [SentenceFaithfulness(sentence="Valve V-2201A caused it.", entailed=True, best_claim_id="c2", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)

    assert len(attributions[0].graph_edge_descriptions) == 1


def test_build_handles_multiple_sentences_independently():
    claim = _claim("c1")
    claim_graph = ClaimGraph(claims=[claim], relations=[])
    scores = [
        SentenceFaithfulness(sentence="Pump P-101 is running.", entailed=True, best_claim_id="c1", score=0.9),
        SentenceFaithfulness(sentence="Unrelated.", entailed=False, best_claim_id=None, score=0.0),
    ]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)

    assert len(attributions) == 2
    assert attributions[0].supporting_claim_ids == ["c1"]
    assert attributions[1].supporting_claim_ids == []


def test_confidence_prefers_propagated_confidence_when_present():
    import dataclasses

    claim = dataclasses.replace(_claim("c1"), propagated_confidence=0.88)
    claim_graph = ClaimGraph(claims=[claim], relations=[])
    scores = [SentenceFaithfulness(sentence="Pump P-101 is running.", entailed=True, best_claim_id="c1", score=0.9)]
    builder = EvidenceAttributionBuilder(InMemoryGraphRepository())

    attributions = builder.build(scores, claim_graph)

    assert attributions[0].confidence == 0.88
