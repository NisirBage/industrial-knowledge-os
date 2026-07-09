from app.entity_resolution.similarity import (
    cosine_similarity,
    edit_distance,
    edit_distance_similarity,
    expand_acronym,
    neighborhood_similarity,
    ontology_similarity,
)
from app.ontology.labels import EntityLabel


def test_edit_distance_of_identical_strings_is_zero():
    assert edit_distance("P-101", "P-101") == 0


def test_edit_distance_counts_a_single_substitution():
    assert edit_distance("P-101", "P-102") == 1


def test_edit_distance_counts_a_single_insertion():
    assert edit_distance("P101", "P-101") == 1


def test_edit_distance_is_case_insensitive():
    assert edit_distance("Pump", "PUMP") == 0


def test_edit_distance_similarity_identical_is_one():
    assert edit_distance_similarity("P-101", "P-101") == 1.0


def test_edit_distance_similarity_close_strings_scores_high():
    assert edit_distance_similarity("P-101", "P-102") > 0.7


def test_edit_distance_similarity_unrelated_strings_scores_low():
    assert edit_distance_similarity("P-101", "Compliance Manual") < 0.3


def test_edit_distance_similarity_both_empty_is_one():
    assert edit_distance_similarity("", "") == 1.0


def test_cosine_similarity_identical_vectors_is_one():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_a_division_error():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_ontology_similarity_same_label_is_one():
    assert ontology_similarity(EntityLabel.PUMP, EntityLabel.PUMP) == 1.0


def test_ontology_similarity_siblings_under_rotating_equipment_is_partial():
    score = ontology_similarity(EntityLabel.PUMP, EntityLabel.MOTOR)
    assert 0.0 < score < 1.0


def test_ontology_similarity_unrelated_branches_is_zero():
    assert ontology_similarity(EntityLabel.PUMP, EntityLabel.STANDARD) == 0.0


def test_ontology_similarity_closer_relatives_score_higher_than_distant_ones():
    close = ontology_similarity(EntityLabel.PUMP, EntityLabel.MOTOR)  # siblings under RotatingEquipment
    distant = ontology_similarity(EntityLabel.PUMP, EntityLabel.SENSOR)  # siblings only under IndustrialAsset
    assert close > distant


def test_neighborhood_similarity_identical_neighbor_sets_is_one():
    assert neighborhood_similarity({"a", "b"}, {"a", "b"}) == 1.0


def test_neighborhood_similarity_disjoint_sets_is_zero():
    assert neighborhood_similarity({"a"}, {"b"}) == 0.0


def test_neighborhood_similarity_both_empty_is_zero_not_undefined():
    assert neighborhood_similarity(set(), set()) == 0.0


def test_neighborhood_similarity_partial_overlap():
    score = neighborhood_similarity({"a", "b", "c"}, {"b", "c", "d"})
    assert score == 0.5  # 2 shared / 4 union


def test_expand_acronym_known_prefix():
    assert expand_acronym("pump") == "P"
    assert expand_acronym("Valve") == "V"


def test_expand_acronym_unknown_word_returns_none():
    assert expand_acronym("widget") is None
