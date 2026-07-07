from app.research.datasets.generator import SyntheticDatasetGenerator


def test_same_seed_produces_identical_documents():
    doc_a = SyntheticDatasetGenerator(seed=1).generate_maintenance_log()
    doc_b = SyntheticDatasetGenerator(seed=1).generate_maintenance_log()
    assert doc_a.text == doc_b.text
    assert doc_a.doc_id == doc_b.doc_id


def test_different_seeds_produce_different_documents():
    doc_a = SyntheticDatasetGenerator(seed=1).generate_maintenance_log()
    doc_b = SyntheticDatasetGenerator(seed=2).generate_maintenance_log()
    assert doc_a.text != doc_b.text


def test_maintenance_log_ground_truth_matches_text():
    doc = SyntheticDatasetGenerator(seed=5).generate_maintenance_log(equipment_tag="P-101")
    assert "P-101" in doc.text
    assert ("Pump", "P-101") in doc.ground_truth_entities
    assert doc.doc_type == "maintenance_log"


def test_inspection_report_ground_truth_matches_text():
    doc = SyntheticDatasetGenerator(seed=5).generate_inspection_report(equipment_tag="V-200")
    assert "V-200" in doc.text
    assert ("Valve", "V-200") in doc.ground_truth_entities


def test_incident_report_produces_failure_mode_entity_and_relation():
    doc = SyntheticDatasetGenerator(seed=5).generate_incident_report(equipment_tag="M-300")
    assert any(label == "FailureMode" for label, _ in doc.ground_truth_entities)
    assert any(rel == "FAILED_DUE_TO" for _, rel, _ in doc.ground_truth_relations)


def test_sop_references_a_standard():
    doc = SyntheticDatasetGenerator(seed=5).generate_sop()
    assert any(label == "Standard" for label, _ in doc.ground_truth_entities)
    assert any(rel == "USES_STANDARD" for _, rel, _ in doc.ground_truth_relations)


def test_oem_manual_references_manufacturer():
    doc = SyntheticDatasetGenerator(seed=5).generate_oem_manual(equipment_tag="P-500", manufacturer="Acme Industrial")
    assert "Acme Industrial" in doc.text
    assert ("Manufacturer", "Acme Industrial") in doc.ground_truth_entities


def test_failure_report_has_cause_and_corrective_action():
    doc = SyntheticDatasetGenerator(seed=5).generate_failure_report(equipment_tag="C-700")
    relation_types = {rel for _, rel, _ in doc.ground_truth_relations}
    assert "FAILED_DUE_TO" in relation_types
    assert "MITIGATED_BY" in relation_types


def test_permit_has_location():
    doc = SyntheticDatasetGenerator(seed=5).generate_permit()
    assert any(label == "Location" for label, _ in doc.ground_truth_entities)


def test_generate_equipment_history_uses_same_tag_across_documents():
    docs = SyntheticDatasetGenerator(seed=7).generate_equipment_history(equipment_tag="P-999", num_events=4)
    assert len(docs) == 4
    assert all(doc.metadata["equipment_tag"] == "P-999" for doc in docs)


def test_generate_timeline_events_returns_requested_count():
    events = SyntheticDatasetGenerator(seed=9).generate_timeline_events(equipment_tag="P-100", num_events=6)
    assert len(events) == 6
    assert all(e.equipment_tag == "P-100" for e in events)


def test_generate_recommendation_outcome_respects_probability_extremes():
    always_accept = SyntheticDatasetGenerator(seed=1).generate_recommendation_outcome(acceptance_probability=1.0)
    always_reject = SyntheticDatasetGenerator(seed=1).generate_recommendation_outcome(acceptance_probability=0.0)
    assert always_accept.accepted is True
    assert always_reject.accepted is False


def test_generate_dataset_produces_requested_count():
    docs = SyntheticDatasetGenerator(seed=3).generate_dataset(15)
    assert len(docs) == 15


def test_generate_dataset_respects_doc_type_filter():
    docs = SyntheticDatasetGenerator(seed=3).generate_dataset(10, doc_types=["permit"])
    assert all(doc.doc_type == "permit" for doc in docs)
