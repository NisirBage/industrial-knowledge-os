from app.contradiction.detector import ContradictionDetector
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle
from app.retrieval.evidence import Evidence


def _bundle(evidence: list[Evidence]) -> EvidenceBundle:
    return EvidenceBundle(evidence=evidence, graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7)


def test_detects_contradiction_for_shared_tag_with_opposite_states():
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Inspection log: Valve V-2201A is closed.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Maintenance note: Valve V-2201A is open.", confidence=0.7),
    ]

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    assert len(contradictions) == 1
    assert set(contradictions[0].conflicting_evidence) == {"c1", "c2"}
    assert contradictions[0].confidence == 0.7


def test_no_contradiction_without_shared_equipment_tag():
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-2201A is closed.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Valve V-9999Z is open.", confidence=0.7),
    ]

    assert ContradictionDetector().detect(_bundle(evidence)) == []


def test_no_contradiction_when_states_agree():
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-2201A is closed.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Valve V-2201A remains closed per the latest check.", confidence=0.7),
    ]

    assert ContradictionDetector().detect(_bundle(evidence)) == []


def test_detects_contradiction_within_a_single_chunk():
    evidence = [
        Evidence(
            document_id="d1",
            page_number=1,
            chunk_id="c1",
            text="Inspection log states Valve V-2201A was closed, while the maintenance record states V-2201A was open.",
            confidence=0.8,
        ),
    ]

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    assert len(contradictions) == 1
    assert contradictions[0].conflicting_evidence == ["c1"]


def test_severity_scales_with_confidence():
    high = ContradictionDetector()._severity_for(0.9)
    medium = ContradictionDetector()._severity_for(0.6)
    low = ContradictionDetector()._severity_for(0.2)

    assert high == Severity.HIGH
    assert medium == Severity.MEDIUM
    assert low == Severity.LOW


def test_multiple_shared_tags_produce_one_contradiction_each_no_duplicates():
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-101 and valve V-2201A are open.", confidence=0.8),
        Evidence(document_id="d2", page_number=1, chunk_id="c2", text="Pump P-101 and valve V-2201A are closed.", confidence=0.7),
    ]

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    # One shared-tag contradiction per equipment tag (P-101, V-2201A), each unique.
    assert len(contradictions) == 2
    tags_mentioned = {c.description.rsplit(" ", 1)[-1] for c in contradictions}
    assert tags_mentioned == {"P-101", "V-2201A"}
