from app.claims.extraction import ClaimExtractionEngine
from app.claims.models import ClaimType
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.reasoning.graph_models import ReasoningEdge, ReasoningGraph, ReasoningNode, ReasoningNodeType
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.retrieval.evidence import Evidence


def _engine() -> ClaimExtractionEngine:
    return ClaimExtractionEngine(
        entity_extraction_service=EntityExtractionService(primary=RegexEntityExtractor()),
        relation_extractor=RuleBasedRelationExtractor(),
    )


def _evidence(text: str, chunk_id: str = "c1") -> Evidence:
    return Evidence(document_id="d1", page_number=1, chunk_id=chunk_id, text=text)


def test_two_entities_and_causal_relation_produce_the_worked_example():
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101 failed due to Valve V-2201A.")])

    fact_claims = graph.claims_of_type(ClaimType.FACT)
    assert len(fact_claims) == 2
    subjects = {c.subject for c in fact_claims}
    assert "P-101" in subjects
    assert "V-2201A" in subjects

    assert len(graph.relations) == 1
    relation = graph.relations[0]
    assert relation.relation_type == "FAILED_DUE_TO"
    claim_ids = {c.id for c in fact_claims}
    assert relation.source_claim_id in claim_ids
    assert relation.target_claim_id in claim_ids


def test_every_claim_carries_source_span_and_evidence_id():
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101 is running normally.")])

    assert len(graph.claims) == 1
    claim = graph.claims[0]
    assert claim.source_evidence_ids == ["c1"]
    assert len(claim.source_spans) == 1
    span = claim.source_spans[0]
    assert span.chunk_id == "c1"
    assert span.document_id == "d1"
    assert span.end_char > span.start_char


def test_no_entities_produces_no_claims():
    engine = _engine()
    graph = engine.extract([_evidence("The weather was clear that day.")])
    assert graph.claims == []
    assert graph.relations == []


def test_multiple_evidence_items_accumulate_into_one_graph():
    engine = _engine()
    graph = engine.extract(
        [
            _evidence("Pump P-101 is running normally.", chunk_id="c1"),
            _evidence("Valve V-2201A was inspected by J. Smith.", chunk_id="c2"),
        ]
    )
    chunk_ids = {span.chunk_id for claim in graph.claims for span in claim.source_spans}
    assert chunk_ids == {"c1", "c2"}


def test_reasoning_graph_risk_and_hypothesis_nodes_become_claims_linked_to_facts():
    engine = _engine()
    evidence = [_evidence("Pump P-101 failed due to Valve V-2201A.", chunk_id="c1")]

    reasoning_graph = ReasoningGraph(
        nodes=[
            ReasoningNode(
                id="risk:c1",
                node_type=ReasoningNodeType.RISK,
                description="Recurring risk: Pump P-101 FAILED_DUE_TO Valve V-2201A",
                confidence=0.75,
                supporting_evidence_ids=["c1"],
            ),
            ReasoningNode(
                id="hypothesis:0",
                node_type=ReasoningNodeType.HYPOTHESIS,
                description="Evidence conflicts on P-101",
                confidence=0.3,
                supporting_evidence_ids=["c1"],
            ),
        ],
        edges=[ReasoningEdge(source_id="risk:c1", target_id="hypothesis:0", relationship="relates_to")],
        summary="",
    )

    graph = engine.extract(evidence, reasoning_graph=reasoning_graph)

    risk_claims = graph.claims_of_type(ClaimType.RISK)
    hypothesis_claims = graph.claims_of_type(ClaimType.HYPOTHESIS)
    assert len(risk_claims) == 1
    assert len(hypothesis_claims) == 1
    assert risk_claims[0].confidence == 0.75
    assert risk_claims[0].source_evidence_ids == ["c1"]

    supports_relations = [r for r in graph.relations if r.relation_type == "SUPPORTS"]
    # 2 fact claims (P-101, V-2201A) each SUPPORTS both the risk and hypothesis claim
    assert len(supports_relations) == 4
    targets = {r.target_claim_id for r in supports_relations}
    assert targets == {risk_claims[0].id, hypothesis_claims[0].id}


def test_three_entities_in_one_sentence_without_cue_produce_no_relations():
    """Mirrors RuleBasedRelationExtractor's own noise-avoidance rule: 3+
    entities in one sentence with no keyword cue are left unlinked."""
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101, Valve V-2201A, and Motor M-30 are all in Zone 4.")])

    fact_claims = graph.claims_of_type(ClaimType.FACT)
    assert len(fact_claims) == 3
    assert graph.relations == []


def test_standard_reference_produces_uses_standard_relation_claim():
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101 operates per ISO 9001.")])

    assert len(graph.relations) == 1
    assert graph.relations[0].relation_type == "USES_STANDARD"


def test_entity_confidence_is_carried_through_to_the_fact_claim():
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101 is running normally.")])

    assert len(graph.claims) == 1
    claim = graph.claims[0]
    assert claim.confidence > 0.0


def test_extract_with_no_reasoning_graph_produces_no_risk_or_hypothesis_claims():
    engine = _engine()
    graph = engine.extract([_evidence("Pump P-101 failed due to Valve V-2201A.")])
    assert graph.claims_of_type(ClaimType.RISK) == []
    assert graph.claims_of_type(ClaimType.HYPOTHESIS) == []
