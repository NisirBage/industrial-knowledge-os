from app.context_compression.engine import CompressionLevel, ContextCompressionEngine
from app.retrieval.evidence import Evidence


def _fusion(chunk_id, text, confidence=0.7):
    return Evidence(document_id="d", page_number=1, chunk_id=chunk_id, text=text, source="fusion", confidence=confidence)


def test_empty_evidence_produces_an_empty_bundle():
    engine = ContextCompressionEngine()

    bundle = engine.compress([])

    assert bundle.compressed_evidence == ()
    assert bundle.stats.original_count == 0
    assert bundle.stats.compression_ratio == 0.0


def test_exact_duplicates_are_removed_and_counted():
    items = [_fusion("c1", "Pump P-101 failed"), _fusion("c2", "Pump P-101 failed"), _fusion("c3", "Valve V-202 leaking")]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert bundle.stats.exact_duplicate_count == 1
    assert bundle.stats.original_count == 3
    assert bundle.stats.compressed_count <= 2


def test_compression_ratio_reflects_real_reduction():
    items = [_fusion(f"c{i}", "Pump P-101 failed due to bearing wear") for i in range(5)]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert bundle.stats.original_count == 5
    assert bundle.stats.compressed_count < 5
    assert bundle.stats.compression_ratio < 1.0


def test_no_reduction_when_every_item_is_genuinely_unique():
    items = [_fusion(f"c{i}", f"Completely distinct unrelated topic number {i} zz{i}yy{i}xx{i}") for i in range(3)]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert bundle.stats.exact_duplicate_count == 0
    assert bundle.stats.compressed_count == 3


def test_summary_string_mentions_real_counts():
    items = [_fusion("c1", "Pump P-101 failed"), _fusion("c2", "Pump P-101 failed")]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert str(bundle.stats.original_count) in bundle.stats.summary
    assert "Compressed" in bundle.stats.summary


def test_aggressive_level_compresses_at_least_as_much_as_light():
    items = [_fusion(f"c{i}", f"Pump P-101 failed due to bearing wear variant {i}") for i in range(6)]
    engine = ContextCompressionEngine()

    light_bundle = engine.compress(items, level=CompressionLevel.LIGHT)
    aggressive_bundle = engine.compress(items, level=CompressionLevel.AGGRESSIVE)

    assert aggressive_bundle.stats.compressed_count <= light_bundle.stats.compressed_count


def test_token_reduction_ratio_is_non_negative_and_at_most_one():
    items = [_fusion(f"c{i}", "Pump P-101 failed due to bearing wear") for i in range(4)]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert 0.0 <= bundle.stats.token_reduction_ratio <= 1.0


def test_compressed_evidence_never_exceeds_original_count():
    items = [_fusion(f"c{i}", f"unique text {i}") for i in range(4)]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert len(bundle.compressed_evidence) <= len(items)


def test_clusters_and_compressed_evidence_are_consistent():
    items = [_fusion(f"c{i}", f"unique text {i}") for i in range(4)]
    engine = ContextCompressionEngine()

    bundle = engine.compress(items)

    assert len(bundle.clusters) == len(bundle.compressed_evidence)
    assert bundle.stats.cluster_count == len(bundle.clusters)
