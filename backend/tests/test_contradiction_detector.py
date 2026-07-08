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


def test_unrelated_entities_in_the_same_sentence_do_not_cross_contaminate():
    """Two different tags, each consistently described with a single state
    across all evidence, must not be flagged — regardless of how many
    other (also non-conflicting) entities appear nearby."""
    evidence = [
        Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Pump P-300 is running normally.", confidence=0.8),
        Evidence(document_id="d1", page_number=1, chunk_id="c2", text="Valve V-400 is closed as expected.", confidence=0.8),
        Evidence(document_id="d1", page_number=1, chunk_id="c3", text="Motor M-500 passed its inspection.", confidence=0.8),
        Evidence(document_id="d1", page_number=1, chunk_id="c4", text="Sensor S-600 is functional.", confidence=0.8),
    ]

    assert ContradictionDetector().detect(_bundle(evidence)) == []


def test_a_sentence_mentioning_two_entities_with_different_states_does_not_conflate_them():
    """A single sentence naming two different tags with two different
    (non-antonym-paired) states should not manufacture a contradiction
    between them — each tag only has one term associated, on its own."""
    evidence = [
        Evidence(
            document_id="d1", page_number=1, chunk_id="c1",
            text="Pump P-101 is running while Valve V-2201A remains closed for isolation.", confidence=0.8,
        ),
    ]

    assert ContradictionDetector().detect(_bundle(evidence)) == []


def test_repeated_duplicate_evidence_produces_exactly_one_contradiction():
    """Regression test for the audit finding: re-ingesting the same two
    conflicting statements many times (e.g. duplicate document upload) must
    not multiply the reported contradiction count — one real disagreement
    is one Contradiction, with every supporting chunk aggregated into it."""
    evidence = []
    for i in range(10):
        evidence.append(
            Evidence(document_id=f"d{i}a", page_number=1, chunk_id=f"c{i}a", text="Valve V-2201A is open.", confidence=0.8)
        )
        evidence.append(
            Evidence(document_id=f"d{i}b", page_number=1, chunk_id=f"c{i}b", text="Valve V-2201A is closed.", confidence=0.7)
        )

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    assert len(contradictions) == 1
    assert len(contradictions[0].conflicting_evidence) == 20


def test_large_evidence_set_with_one_real_conflict_stays_bounded():
    """A large, mostly-agreeing evidence set with exactly one genuine
    disagreement embedded in it must report exactly one contradiction, not
    one per pairwise combination — the combinatorial-explosion bug found
    during the v1.0.0 audit."""
    evidence = [
        Evidence(document_id=f"d{i}", page_number=1, chunk_id=f"c{i}", text=f"Equipment P-{100 + i} is operational.", confidence=0.8)
        for i in range(30)
    ]
    evidence.append(Evidence(document_id="dx", page_number=1, chunk_id="cx1", text="Valve V-2201A is open.", confidence=0.8))
    evidence.append(Evidence(document_id="dy", page_number=1, chunk_id="cx2", text="Valve V-2201A is closed.", confidence=0.7))

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    assert len(contradictions) == 1
    assert set(contradictions[0].conflicting_evidence) == {"cx1", "cx2"}


def test_demo_dataset_style_cross_document_conflict_is_detected_once():
    """Reproduces the real demo_data/ scenario that originally surfaced the
    combinatorial false-positive bug: an inspection report and an incident
    report disagreeing about Valve V-2201A's state, alongside several other
    real sentences from the same documents that mention P-101/V-2201A in
    non-conflicting contexts."""
    evidence = [
        Evidence(
            document_id="d1", page_number=1, chunk_id="c1",
            text=(
                "Valve V-2201A was found in the OPEN position at the time of inspection, "
                "consistent with normal operating configuration."
            ),
            confidence=0.8,
        ),
        Evidence(
            document_id="d2", page_number=1, chunk_id="c2",
            text=(
                "Root cause investigation determined that Valve V-2201A, the upstream isolation valve, "
                "was found in the CLOSED position at the time of the trip."
            ),
            confidence=0.7,
        ),
        Evidence(
            document_id="d3", page_number=1, chunk_id="c3",
            text="Equipment Tag: P-101. Type: Centrifugal Transfer Pump. Manufacturer: Sundyne.",
            confidence=0.8,
        ),
        Evidence(
            document_id="d4", page_number=1, chunk_id="c4",
            text="Confirm Valve V-2201A is fully OPEN before requesting permission to start P-101.",
            confidence=0.8,
        ),
    ]

    contradictions = ContradictionDetector().detect(_bundle(evidence))

    assert len(contradictions) == 1
    assert contradictions[0].description == "Conflicting state reported for V-2201A"
    assert set(contradictions[0].conflicting_evidence) == {"c1", "c2", "c4"}
