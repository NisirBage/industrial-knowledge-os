from app.context_compression.dedup import (
    entity_overlap,
    near_duplicate_score,
    normalized_text_hash,
    remove_exact_duplicates,
    remove_near_duplicates,
    text_overlap,
)
from app.retrieval.evidence import Evidence


def _evidence(chunk_id: str, text: str, entity_ids: list[str] | None = None) -> Evidence:
    return Evidence(document_id="d", page_number=1, chunk_id=chunk_id, text=text, entity_ids=entity_ids or [], confidence=0.7)


def test_normalized_text_hash_ignores_case_and_extra_whitespace():
    assert normalized_text_hash("Pump  P-101  failed") == normalized_text_hash("pump p-101 failed")


def test_normalized_text_hash_differs_for_different_text():
    assert normalized_text_hash("Pump P-101 failed") != normalized_text_hash("Valve V-202 failed")


def test_text_overlap_identical_text_is_one():
    assert text_overlap("pump p-101 failed", "pump p-101 failed") == 1.0


def test_text_overlap_disjoint_text_is_zero():
    assert text_overlap("alpha beta", "gamma delta") == 0.0


def test_text_overlap_partial():
    assert text_overlap("pump p-101 failed today", "pump p-101 failed yesterday") == 0.6


def test_entity_overlap_shared_entities():
    a = _evidence("c1", "text", entity_ids=["e1", "e2"])
    b = _evidence("c2", "text", entity_ids=["e2", "e3"])
    assert entity_overlap(a, b) == round(1 / 3, 4)


def test_entity_overlap_no_entities_is_zero():
    a = _evidence("c1", "text")
    b = _evidence("c2", "text")
    assert entity_overlap(a, b) == 0.0


def test_near_duplicate_score_uses_text_and_entity_signals_without_an_embedder():
    a = _evidence("c1", "Pump P-101 failed due to bearing wear", entity_ids=["e1"])
    b = _evidence("c2", "Pump P-101 failed due to bearing wear", entity_ids=["e1"])
    assert near_duplicate_score(a, b) == 1.0


def test_near_duplicate_score_includes_embedding_signal_when_embedder_given():
    class _ConstantEmbedder:
        dimension = 2

        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    a = _evidence("c1", "alpha")
    b = _evidence("c2", "beta")
    score_without = near_duplicate_score(a, b)
    score_with = near_duplicate_score(a, b, embedder=_ConstantEmbedder())
    assert score_with != score_without  # the embedding signal (1.0, since the fake embedder is constant) changes the average


def test_remove_exact_duplicates_keeps_one_representative_per_hash():
    items = [_evidence("c1", "Pump P-101 failed"), _evidence("c2", "Pump P-101 failed"), _evidence("c3", "Valve V-202 leaking")]

    deduped, count = remove_exact_duplicates(items)

    assert len(deduped) == 2
    assert count == 1


def test_remove_exact_duplicates_never_drops_unique_evidence():
    items = [_evidence(f"c{i}", f"unique text number {i}") for i in range(5)]

    deduped, count = remove_exact_duplicates(items)

    assert len(deduped) == 5
    assert count == 0


def test_remove_near_duplicates_merges_highly_similar_text():
    items = [
        _evidence("c1", "Pump P-101 failed due to bearing wear", entity_ids=["e1"]),
        _evidence("c2", "Pump P-101 failed due to bearing wear issue", entity_ids=["e1"]),
        _evidence("c3", "Valve V-202 leaking at flange", entity_ids=["e2"]),
    ]

    deduped, count = remove_near_duplicates(items, threshold=0.6)

    assert len(deduped) == 2
    assert count == 1
    assert deduped[0].chunk_id == "c1"  # first-seen kept


def test_remove_near_duplicates_keeps_all_when_nothing_is_similar():
    items = [_evidence("c1", "alpha bravo charlie"), _evidence("c2", "delta echo foxtrot")]

    deduped, count = remove_near_duplicates(items, threshold=0.85)

    assert len(deduped) == 2
    assert count == 0
