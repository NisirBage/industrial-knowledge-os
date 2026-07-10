from unittest.mock import patch

from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle
from app.nli.contradiction_bridge import NLIContradictionDetector
from app.nli.factory import build_nli_detector
from app.nli.models import NLILabel, NLIResult
from app.nli.rule_based import RuleBasedNLIDetector
from app.nli.transformer_nli import TransformerNLIDetector
from app.retrieval.evidence import Evidence


def test_rule_based_detects_contradiction_from_shared_antonym_vocabulary():
    detector = RuleBasedNLIDetector()
    result = detector.classify("Valve V-2201A is open.", "Valve V-2201A was confirmed closed.")
    assert result.label == NLILabel.CONTRADICTION


def test_rule_based_detects_entailment_from_high_word_overlap():
    detector = RuleBasedNLIDetector()
    result = detector.classify(
        "Pump P-101 failed due to cavitation.",
        "Pump P-101 failed due to cavitation.",
    )
    assert result.label == NLILabel.ENTAILMENT


def test_rule_based_defaults_to_neutral_for_unrelated_sentences():
    detector = RuleBasedNLIDetector()
    result = detector.classify("Pump P-101 is running.", "The inspection schedule was updated in March.")
    assert result.label == NLILabel.NEUTRAL


def test_build_nli_detector_falls_back_to_rule_based_when_model_load_fails():
    with patch("app.nli.transformer_nli.TransformerNLIDetector", side_effect=RuntimeError("no network")):
        detector = build_nli_detector("some/unavailable-model")
    assert isinstance(detector, RuleBasedNLIDetector)


def test_transformer_nli_label_resolution_maps_id2label_to_nli_labels():
    detector = TransformerNLIDetector.__new__(TransformerNLIDetector)
    detector._id2label = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
    assert detector._label_for(0) == NLILabel.CONTRADICTION
    assert detector._label_for(1) == NLILabel.ENTAILMENT
    assert detector._label_for(2) == NLILabel.NEUTRAL


def test_transformer_nli_label_resolution_defaults_to_neutral_for_unknown_index():
    detector = TransformerNLIDetector.__new__(TransformerNLIDetector)
    detector._id2label = {0: "contradiction"}
    assert detector._label_for(99) == NLILabel.NEUTRAL


class _StubNLIDetector:
    """Deterministic NLI stub for contradiction-bridge tests — avoids
    depending on RuleBasedNLIDetector's own heuristic thresholds."""

    def __init__(self, contradicts: set[tuple[str, str]]):
        self._contradicts = contradicts

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        label = NLILabel.CONTRADICTION if (premise, hypothesis) in self._contradicts or (hypothesis, premise) in self._contradicts else NLILabel.NEUTRAL
        return NLIResult(premise, hypothesis, label, 0.9)


def test_nli_contradiction_detector_flags_pair_the_injected_model_calls_contradiction():
    sentence_a = "Valve V-2201A is fully open."
    sentence_b = "Valve V-2201A remains shut."
    stub = _StubNLIDetector(contradicts={(sentence_a, sentence_b)})
    bridge = NLIContradictionDetector(stub)

    bundle = EvidenceBundle(
        evidence=[
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text=sentence_a, confidence=0.9),
            Evidence(document_id="d1", page_number=2, chunk_id="c2", text=sentence_b, confidence=0.8),
        ],
        graph_evidence=[],
        timeline=[],
        metadata=[],
        missing_information=[],
        confidence=0.7,
    )

    contradictions = bridge.detect(bundle)
    assert len(contradictions) == 1
    assert contradictions[0].conflicting_evidence == ["c1", "c2"]
    assert contradictions[0].metadata["detector"] == "nli"


def test_nli_contradiction_detector_returns_nothing_when_model_finds_no_contradiction():
    stub = _StubNLIDetector(contradicts=set())
    bridge = NLIContradictionDetector(stub)
    bundle = EvidenceBundle(
        evidence=[
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-2201A is open.", confidence=0.9),
            Evidence(document_id="d1", page_number=2, chunk_id="c2", text="Valve V-2201A was inspected.", confidence=0.8),
        ],
        graph_evidence=[],
        timeline=[],
        metadata=[],
        missing_information=[],
        confidence=0.7,
    )
    assert bridge.detect(bundle) == []


def test_nli_contradiction_detector_matches_contract_of_rule_based_detector():
    """Both detectors must be swappable behind the same `.detect(bundle)
    -> list[Contradiction]` contract — this exercises the exact substitution
    ReasoningAgent's contradiction_detector collaborator relies on."""
    from app.contradiction.detector import ContradictionDetector

    rule_based = ContradictionDetector()
    nli_based = NLIContradictionDetector(RuleBasedNLIDetector())

    bundle = EvidenceBundle(
        evidence=[Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-2201A is open.")],
        graph_evidence=[],
        timeline=[],
        metadata=[],
        missing_information=[],
        confidence=0.7,
    )

    result_rule = rule_based.detect(bundle)
    result_nli = nli_based.detect(bundle)
    assert isinstance(result_rule, list)
    assert isinstance(result_nli, list)
    assert all(isinstance(c, Contradiction) for c in result_rule + result_nli)


def test_nli_contradiction_detector_deduplicates_same_pair_across_sentence_splits():
    sentence_a = "Valve V-2201A is fully open."
    sentence_b = "Valve V-2201A remains shut."
    stub = _StubNLIDetector(contradicts={(sentence_a, sentence_b)})
    bridge = NLIContradictionDetector(stub)

    bundle = EvidenceBundle(
        evidence=[
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text=f"{sentence_a} Some other unrelated sentence.", confidence=0.9),
            Evidence(document_id="d1", page_number=2, chunk_id="c2", text=sentence_b, confidence=0.8),
        ],
        graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7,
    )

    contradictions = bridge.detect(bundle)
    assert len(contradictions) == 1


def test_nli_contradiction_detector_ignores_pairs_within_the_same_chunk():
    stub = _StubNLIDetector(contradicts={("Valve V-2201A is open.", "Valve V-2201A is closed.")})
    bridge = NLIContradictionDetector(stub)
    bundle = EvidenceBundle(
        evidence=[
            Evidence(document_id="d1", page_number=1, chunk_id="c1", text="Valve V-2201A is open. Valve V-2201A is closed.", confidence=0.9),
        ],
        graph_evidence=[], timeline=[], metadata=[], missing_information=[], confidence=0.7,
    )

    assert bridge.detect(bundle) == []


def test_rule_based_nli_is_case_insensitive_for_antonym_matching():
    detector = RuleBasedNLIDetector()
    result = detector.classify("Valve V-2201A is OPEN.", "Valve V-2201A was CLOSED.")
    assert result.label == NLILabel.CONTRADICTION


def test_transformer_nli_falls_back_to_standard_label_order_when_config_unavailable():
    detector = TransformerNLIDetector.__new__(TransformerNLIDetector)
    assert detector._resolve_id2label() == {0: "contradiction", 1: "entailment", 2: "neutral"}


def test_severity_thresholds_match_contradiction_confidence():
    stub = _StubNLIDetector(contradicts={("a", "b")})
    bridge = NLIContradictionDetector(stub)
    assert bridge._severity_for(0.9) == Severity.HIGH
    assert bridge._severity_for(0.6) == Severity.MEDIUM
    assert bridge._severity_for(0.1) == Severity.LOW
