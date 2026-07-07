from app.documents.chunking import Chunk
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.ontology.labels import EntityLabel


def _chunk(text: str) -> Chunk:
    return Chunk(id="c1", document_id="d1", text=text, page_number=1, chunk_index=0)


def test_regex_extractor_finds_equipment_tags_and_standards():
    chunk = _chunk("Pump P-101 failed inspection under ISO 9001 and OISD-105 requirements.")

    entities = RegexEntityExtractor().extract(chunk)
    labels_values = {(e.label, e.canonical_name) for e in entities}

    assert (EntityLabel.PUMP, "P-101") in labels_values
    assert any(label == EntityLabel.STANDARD and "ISO" in value for label, value in labels_values)
    assert any(label == EntityLabel.REGULATION and "OISD" in value.upper() for label, value in labels_values)


def test_regex_extractor_empty_text_returns_no_entities():
    assert RegexEntityExtractor().extract(_chunk("")) == []


def test_regex_extractor_maps_tag_prefix_to_equipment_type():
    chunk = _chunk("Valve V-2201A is upstream of Motor M-30.")
    entities = RegexEntityExtractor().extract(chunk)
    labels_values = {(e.label, e.canonical_name) for e in entities}

    assert (EntityLabel.VALVE, "V-2201A") in labels_values
    assert (EntityLabel.MOTOR, "M-30") in labels_values


def test_entity_extraction_service_deduplicates_same_entity_across_extractors():
    chunk = _chunk("Pump P-101 failed. P-101 requires quarterly lubrication.")
    service = EntityExtractionService(primary=RegexEntityExtractor())

    entities = service.extract(chunk)
    pump_entities = [e for e in entities if e.canonical_name == "P-101"]

    assert len(pump_entities) == 1
    assert pump_entities[0].label == EntityLabel.PUMP
