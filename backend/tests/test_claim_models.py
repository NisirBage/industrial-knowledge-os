import dataclasses

from app.claims.models import (
    Claim,
    ClaimGraph,
    ClaimRelation,
    ClaimSpan,
    ClaimType,
    ClaimVerificationStatus,
)


def _claim(claim_id: str, claim_type: ClaimType = ClaimType.FACT) -> Claim:
    return Claim(
        id=claim_id,
        claim_type=claim_type,
        text=f"text for {claim_id}",
        subject="Pump P-101",
        predicate="failure_mode",
        object="cavitation",
        confidence=0.8,
        source_evidence_ids=["chunk-1"],
        source_spans=[ClaimSpan(chunk_id="chunk-1", document_id="doc-1", start_char=0, end_char=10)],
    )


def test_claim_defaults_are_unverified_and_unpropagated():
    claim = _claim("c1")
    assert claim.verification_status == ClaimVerificationStatus.UNKNOWN
    assert claim.propagated_confidence is None
    assert claim.support_count == 0
    assert claim.contradiction_count == 0


def test_claim_graph_add_and_lookup():
    graph = ClaimGraph(claims=[], relations=[])
    graph.add_claim(_claim("c1"))
    graph.add_claim(_claim("c2", claim_type=ClaimType.RISK))
    graph.add_relation(ClaimRelation(id="r1", source_claim_id="c1", target_claim_id="c2", relation_type="CAUSES", confidence=0.7))

    assert graph.claim_by_id("c1") is not None
    assert graph.claim_by_id("missing") is None
    assert len(graph.claims_of_type(ClaimType.FACT)) == 1
    assert len(graph.claims_of_type(ClaimType.RISK)) == 1
    assert [r.id for r in graph.relations_from("c1")] == ["r1"]
    assert [r.id for r in graph.relations_to("c2")] == ["r1"]
    assert graph.relations_from("c2") == []


def test_claim_span_fields():
    span = ClaimSpan(chunk_id="c1", document_id="d1", start_char=5, end_char=15)
    assert span.chunk_id == "c1"
    assert span.document_id == "d1"
    assert span.end_char - span.start_char == 10


def test_relations_to_and_from_on_empty_graph_return_empty_lists():
    graph = ClaimGraph(claims=[], relations=[])
    assert graph.relations_from("missing") == []
    assert graph.relations_to("missing") == []


def test_claim_graph_with_claims_returns_new_graph_same_relations():
    original = ClaimGraph(claims=[_claim("c1")], relations=[ClaimRelation(id="r1", source_claim_id="c1", target_claim_id="c1", relation_type="RELATED_TO")])
    updated_claim = dataclasses.replace(original.claims[0], verification_status=ClaimVerificationStatus.SUPPORTED)

    new_graph = original.with_claims([updated_claim])

    assert new_graph is not original
    assert new_graph.claims[0].verification_status == ClaimVerificationStatus.SUPPORTED
    assert original.claims[0].verification_status == ClaimVerificationStatus.UNKNOWN
    assert new_graph.relations == original.relations
    assert new_graph.relations is not original.relations
