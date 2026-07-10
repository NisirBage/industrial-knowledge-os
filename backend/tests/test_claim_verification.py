import dataclasses

from app.claims.models import Claim, ClaimGraph, ClaimRelation, ClaimType, ClaimVerificationStatus
from app.claims.verification import ClaimVerificationEngine
from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle, TimelineEvent
from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import InMemoryGraphRepository
from app.retrieval.evidence import Evidence


def _fact_claim(claim_id: str, subject: str, chunk_id: str = "c1", object_: str = "pump", confidence: float = 0.6) -> Claim:
    return Claim(
        id=claim_id,
        claim_type=ClaimType.FACT,
        text=f"Equipment = {subject}",
        subject=subject,
        predicate="is_a",
        object=object_,
        confidence=confidence,
        source_evidence_ids=[chunk_id],
    )


def _bundle(evidence: list[Evidence], contradictions: list[Contradiction] | None = None, timeline: list[TimelineEvent] | None = None) -> EvidenceBundle:
    return EvidenceBundle(
        evidence=evidence,
        graph_evidence=[],
        timeline=timeline or [],
        metadata=[],
        missing_information=[],
        confidence=0.7,
        contradictions=contradictions or [],
    )


def test_supported_when_graph_entry_and_corroborating_evidence_both_exist():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle(
        [
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running."),
            Evidence(document_id="d1", page_number=2, chunk_id="c2", text="P-101 was inspected last week."),
        ]
    )

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.SUPPORTED
    assert result.support_count == 1


def test_partially_supported_when_only_graph_entry_exists():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle([Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running.")])

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.PARTIALLY_SUPPORTED


def test_partially_supported_when_only_corroborating_evidence_exists_without_graph_entry():
    repo = InMemoryGraphRepository()  # no node created — not in the graph
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle(
        [
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running."),
            Evidence(document_id="d1", page_number=2, chunk_id="c2", text="P-101 needs maintenance."),
        ]
    )

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.PARTIALLY_SUPPORTED


def test_insufficient_evidence_when_neither_graph_nor_corroboration_exists():
    repo = InMemoryGraphRepository()
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle([Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running.")])

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE


def test_contradicted_when_origin_chunk_is_in_a_conflicting_evidence_list():
    repo = InMemoryGraphRepository()
    repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    contradiction = Contradiction(
        conflicting_evidence=["c1", "c2"],
        confidence=0.9,
        severity=Severity.HIGH,
        resolution_strategy="Verify manually.",
        recommended_clarification="Conflicting state for P-101.",
    )
    bundle = _bundle(
        [Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 is running.")],
        contradictions=[contradiction],
    )

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.CONTRADICTED
    assert result.contradiction_count == 1


def test_recommendation_and_prediction_claims_are_left_unknown_not_verified_here():
    repo = InMemoryGraphRepository()
    engine = ClaimVerificationEngine(repo)

    claim = Claim(id="c1", claim_type=ClaimType.RECOMMENDATION, text="Investigate P-101", subject="P-101", predicate="recommend")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle([])

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.UNKNOWN


def test_drop_unsupported_removes_contradicted_insufficient_and_unknown_but_keeps_others():
    supported = dataclasses.replace(_fact_claim("supported", "P-101"), verification_status=ClaimVerificationStatus.SUPPORTED)
    partial = dataclasses.replace(_fact_claim("partial", "V-2201A"), verification_status=ClaimVerificationStatus.PARTIALLY_SUPPORTED)
    contradicted = dataclasses.replace(_fact_claim("contradicted", "M-30"), verification_status=ClaimVerificationStatus.CONTRADICTED)
    insufficient = dataclasses.replace(_fact_claim("insufficient", "T-1"), verification_status=ClaimVerificationStatus.INSUFFICIENT_EVIDENCE)

    claim_graph = ClaimGraph(
        claims=[supported, partial, contradicted, insufficient],
        relations=[
            ClaimRelation(id="r1", source_claim_id="supported", target_claim_id="partial", relation_type="RELATED_TO"),
            ClaimRelation(id="r2", source_claim_id="supported", target_claim_id="contradicted", relation_type="RELATED_TO"),
        ],
    )
    repo = InMemoryGraphRepository()
    engine = ClaimVerificationEngine(repo)

    kept = engine.drop_unsupported(claim_graph)

    kept_ids = {c.id for c in kept.claims}
    assert kept_ids == {"supported", "partial"}
    assert [r.id for r in kept.relations] == ["r1"]


def test_graph_repository_exception_is_treated_as_no_graph_support():
    class _ExplodingGraphRepository(InMemoryGraphRepository):
        def find_entity(self, canonical_name, label=None):
            raise RuntimeError("simulated backend failure")

    engine = ClaimVerificationEngine(_ExplodingGraphRepository())
    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle([Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running.")])

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.verification_status == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE


def test_relation_with_unrecognized_object_label_is_not_confirmed_but_does_not_crash():
    repo = InMemoryGraphRepository()
    engine = ClaimVerificationEngine(repo)

    source = _fact_claim("c-source", "P-101", object_="not_a_real_label")
    target = _fact_claim("c-target", "V-2201A")
    graph = ClaimGraph(
        claims=[source, target],
        relations=[ClaimRelation(id="rel1", source_claim_id="c-source", target_claim_id="c-target", relation_type="FAILED_DUE_TO")],
    )
    bundle = _bundle([])

    verified = engine.verify(graph, bundle)
    assert verified.claim_by_id("c-source").verification_status == ClaimVerificationStatus.INSUFFICIENT_EVIDENCE


def test_timeline_events_count_as_corroboration():
    repo = InMemoryGraphRepository()
    engine = ClaimVerificationEngine(repo)

    claim = _fact_claim("c1", "P-101")
    graph = ClaimGraph(claims=[claim], relations=[])
    bundle = _bundle(
        [Evidence(document_id="d", page_number=1, chunk_id="c1", text="Pump P-101 is running.")],
        timeline=[TimelineEvent(description="P-101 was inspected in March.", timestamp="2024-03-01", source_chunk_id="c2", confidence=0.8)],
    )

    verified = engine.verify(graph, bundle)
    result = verified.claim_by_id("c1")
    assert result.support_count >= 1


def test_confirmed_ontology_relation_boosts_graph_support_for_both_endpoints():
    repo = InMemoryGraphRepository()
    source_id = repo.create_node(EntityLabel.PUMP, "P-101", confidence=0.9)
    target_id = repo.create_node(EntityLabel.VALVE, "V-2201A", confidence=0.9)
    repo.create_relation(source_id, target_id, RelationType.FAILED_DUE_TO, confidence=0.8)
    engine = ClaimVerificationEngine(repo)

    source_claim = _fact_claim("c-source", "P-101", object_=EntityLabel.PUMP.value)
    target_claim = _fact_claim("c-target", "V-2201A", object_=EntityLabel.VALVE.value)
    claim_graph = ClaimGraph(
        claims=[source_claim, target_claim],
        relations=[
            ClaimRelation(
                id="rel1",
                source_claim_id="c-source",
                target_claim_id="c-target",
                relation_type=RelationType.FAILED_DUE_TO.value,
                confidence=0.7,
            )
        ],
    )
    bundle = _bundle([])

    verified = engine.verify(claim_graph, bundle)
    source_result = verified.claim_by_id("c-source")
    target_result = verified.claim_by_id("c-target")
    # Both endpoints get at least a PARTIALLY_SUPPORTED status purely from
    # the confirmed relation, even with no other evidence/graph entity checks matching.
    assert source_result.verification_status in {ClaimVerificationStatus.SUPPORTED, ClaimVerificationStatus.PARTIALLY_SUPPORTED}
    assert target_result.verification_status in {ClaimVerificationStatus.SUPPORTED, ClaimVerificationStatus.PARTIALLY_SUPPORTED}
    assert source_result.support_count >= 1
    assert target_result.support_count >= 1
