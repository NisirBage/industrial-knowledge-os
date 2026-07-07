from app.documents.chunking import Chunk
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.ontology.labels import RelationType
from app.relation_extraction.rule_based import RuleBasedRelationExtractor


def _extract_relations(text: str):
    chunk = Chunk(id="c1", document_id="d1", text=text, page_number=1, chunk_index=0)
    entities = RegexEntityExtractor().extract(chunk)
    relations = RuleBasedRelationExtractor().extract(entities, chunk)
    return entities, relations


def test_keyword_cue_produces_typed_relation():
    entities, relations = _extract_relations("Pump P-101 is connected to Valve V-2201A.")

    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.CONNECTED_TO
    entity_ids = {e.id for e in entities}
    assert relations[0].source_entity_id in entity_ids
    assert relations[0].target_entity_id in entity_ids


def test_two_entities_no_cue_falls_back_to_related_to():
    _entities, relations = _extract_relations("Pump P-101 and Valve V-2201A are shown on the diagram.")

    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.RELATED_TO


def test_standard_reference_produces_uses_standard():
    _entities, relations = _extract_relations("Pump P-101 operates per ISO 9001.")

    assert len(relations) == 1
    assert relations[0].relation_type == RelationType.USES_STANDARD


def test_no_relations_when_fewer_than_two_entities():
    _entities, relations = _extract_relations("Pump P-101 is running normally.")
    assert relations == []


def test_three_entities_in_one_sentence_without_cue_produces_no_noisy_related_to():
    _entities, relations = _extract_relations("Pump P-101, Valve V-2201A, and Motor M-30 are all in Zone 4.")
    # 3+ entities with no keyword cue: left unlinked rather than guessing,
    # since RELATED_TO fallback only applies to sentences with exactly 2 entities.
    assert relations == []
