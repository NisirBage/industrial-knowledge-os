import random

from app.research.datasets.generator import SyntheticDatasetGenerator
from app.research.datasets.models import NoiseConfig
from app.research.datasets.noise import NoiseInjector


def _document():
    return SyntheticDatasetGenerator(seed=1).generate_maintenance_log(equipment_tag="P-101")


def test_ocr_errors_change_text_at_full_rate():
    document = _document()
    noisy = NoiseInjector().apply(document, NoiseConfig(ocr_error_rate=1.0), random.Random(1))
    assert noisy.text != document.text


def test_zero_noise_config_leaves_text_unchanged_except_metadata():
    document = _document()
    noisy = NoiseInjector().apply(document, NoiseConfig(), random.Random(1))
    assert noisy.text == document.text


def test_ground_truth_preserved_through_noise():
    document = _document()
    noisy = NoiseInjector().apply(document, NoiseConfig(ocr_error_rate=0.5, typo_rate=0.5), random.Random(1))
    assert noisy.ground_truth_entities == document.ground_truth_entities
    assert noisy.ground_truth_relations == document.ground_truth_relations


def test_typo_injection_is_deterministic_for_same_seed():
    document = _document()
    noisy_a = NoiseInjector().apply(document, NoiseConfig(typo_rate=1.0), random.Random(42))
    noisy_b = NoiseInjector().apply(document, NoiseConfig(typo_rate=1.0), random.Random(42))
    assert noisy_a.text == noisy_b.text


def test_missing_field_drops_a_sentence():
    document = _document()
    noisy = NoiseInjector().apply(document, NoiseConfig(missing_field_rate=1.0), random.Random(1))
    assert len(noisy.text) < len(document.text)


def test_version_drift_appends_revision_marker():
    document = _document()
    noisy = NoiseInjector().apply(document, NoiseConfig(version_drift=True), random.Random(1))
    assert "Rev." in noisy.text


def test_duplicate_produces_new_id_with_identical_text():
    document = _document()
    duplicate = NoiseInjector().duplicate(document, random.Random(1))
    assert duplicate.doc_id != document.doc_id
    assert duplicate.text == document.text
    assert duplicate.metadata["duplicate_of"] == document.doc_id


def test_generate_conflicting_flips_a_state_word():
    document = SyntheticDatasetGenerator(seed=1).generate_incident_report(equipment_tag="V-1")
    # Force a known state word into the text to make the conflict deterministic to check.
    from dataclasses import replace

    stateful_document = replace(document, text=document.text + " The valve was open at the time.")
    conflicting = NoiseInjector().generate_conflicting(stateful_document, random.Random(1))

    assert "closed" in conflicting.text.lower()
    assert conflicting.metadata["conflicts_with"] == stateful_document.doc_id


def test_generate_conflicting_preserves_ground_truth():
    document = _document()
    conflicting = NoiseInjector().generate_conflicting(document, random.Random(1))
    assert conflicting.ground_truth_entities == document.ground_truth_entities
