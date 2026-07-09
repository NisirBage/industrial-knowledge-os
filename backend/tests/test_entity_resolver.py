from app.entity_resolution.resolver import EntityResolver, KnownEntity
from app.ontology.labels import EntityLabel


def test_resolves_a_close_typo_variant_to_the_existing_entity():
    known = [KnownEntity(node_id="n1", canonical_name="P-101", label=EntityLabel.PUMP)]
    resolver = EntityResolver()

    result = resolver.resolve("P-102", EntityLabel.PUMP, known)

    # P-102 vs P-101 is a single-character edit + same ontology label ->
    # combined average clears the default 0.82 threshold.
    assert result is not None
    assert result.canonical_id == "n1"
    assert result.canonical_name == "P-101"


def test_does_not_resolve_a_completely_unrelated_name():
    known = [KnownEntity(node_id="n1", canonical_name="P-101", label=EntityLabel.PUMP)]
    resolver = EntityResolver()

    result = resolver.resolve("Compliance Manual", EntityLabel.DOCUMENT, known)

    assert result is None


def test_returns_none_with_no_known_entities():
    resolver = EntityResolver()

    assert resolver.resolve("P-101", EntityLabel.PUMP, []) is None


def test_picks_the_best_match_among_multiple_known_entities():
    known = [
        KnownEntity(node_id="far", canonical_name="V-999", label=EntityLabel.VALVE),
        KnownEntity(node_id="close", canonical_name="P-101", label=EntityLabel.PUMP),
    ]
    resolver = EntityResolver()

    result = resolver.resolve("P-101", EntityLabel.PUMP, known)

    assert result.canonical_id == "close"


def test_neighborhood_similarity_signal_is_included_when_neighbor_ids_are_given():
    known = [KnownEntity(node_id="n1", canonical_name="Feedwater Pump", label=EntityLabel.PUMP, neighbor_ids=frozenset({"valve-1", "motor-1"}))]
    resolver = EntityResolver(merge_threshold=0.5)

    result = resolver.resolve("Main Feed Pump", EntityLabel.PUMP, known, candidate_neighbor_ids=frozenset({"valve-1", "motor-1"}))

    assert result is not None
    assert any(s.name == "neighborhood" and s.score == 1.0 for s in result.signals)


def test_embedding_similarity_signal_is_included_when_embedder_and_known_embedding_are_present():
    class _ConstantEmbedder:
        dimension = 2

        def embed_texts(self, texts):
            return [[1.0, 0.0] for _ in texts]

    known = [KnownEntity(node_id="n1", canonical_name="Something Else Entirely", label=EntityLabel.PUMP, embedding=[1.0, 0.0])]
    resolver = EntityResolver(embedder=_ConstantEmbedder(), merge_threshold=0.4)

    result = resolver.resolve("Totally Different Text", EntityLabel.PUMP, known)

    assert result is not None
    assert any(s.name == "embedding" and s.score == 1.0 for s in result.signals)


def test_no_embedder_means_no_embedding_signal_is_fabricated():
    known = [KnownEntity(node_id="n1", canonical_name="P-101", label=EntityLabel.PUMP, embedding=[1.0, 0.0])]
    resolver = EntityResolver(embedder=None)

    resolver.resolve("P-101", EntityLabel.PUMP, known)
    signals = resolver._score("P-101", EntityLabel.PUMP, frozenset(), known[0])

    assert not any(s.name == "embedding" for s in signals)


def test_merge_reason_names_the_strongest_signal():
    known = [KnownEntity(node_id="n1", canonical_name="P-101", label=EntityLabel.PUMP)]
    resolver = EntityResolver()

    result = resolver.resolve("P-101", EntityLabel.PUMP, known)

    assert result.merge_reason.startswith("edit_distance=") or result.merge_reason.startswith("ontology=")


def test_custom_merge_threshold_is_respected():
    known = [KnownEntity(node_id="n1", canonical_name="P-101", label=EntityLabel.PUMP)]
    lenient = EntityResolver(merge_threshold=0.3)
    strict = EntityResolver(merge_threshold=0.99)

    assert lenient.resolve("P-999", EntityLabel.PUMP, known) is not None
    assert strict.resolve("P-999", EntityLabel.PUMP, known) is None
