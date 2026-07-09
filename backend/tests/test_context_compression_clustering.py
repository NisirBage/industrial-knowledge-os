from app.context_compression.clustering import cluster_evidence
from app.retrieval.evidence import Evidence


def _evidence(chunk_id, text, source="fusion", entity_ids=None):
    return Evidence(document_id="d", page_number=1, chunk_id=chunk_id, text=text, source=source, entity_ids=entity_ids or [], confidence=0.7)


class _GroupedEmbedder:
    """Deterministic fake: text starting with "A" embeds near [1,0], text
    starting with "B" embeds near [0,1] — two well-separated real clusters
    without needing a real ML model in a unit test."""

    dimension = 2

    def embed_texts(self, texts):
        return [[1.0, 0.01] if t.startswith("A") else [0.01, 1.0] for t in texts]


def test_empty_evidence_returns_no_clusters():
    assert cluster_evidence([]) == []


def test_single_item_is_its_own_cluster():
    clusters = cluster_evidence([_evidence("c1", "Alpha text")])

    assert len(clusters) == 1
    assert clusters[0].member_count == 1
    assert clusters[0].representative.chunk_id == "c1"


def test_without_an_embedder_every_item_is_a_singleton_cluster_not_a_guess():
    items = [_evidence("c1", "Alpha"), _evidence("c2", "Alpha similar text")]

    clusters = cluster_evidence(items, embedder=None)

    assert len(clusters) == 2


def test_with_an_embedder_similar_items_cluster_together():
    items = [_evidence("c1", "Alpha one"), _evidence("c2", "Alpha two"), _evidence("c3", "Bravo one")]

    clusters = cluster_evidence(items, embedder=_GroupedEmbedder(), distance_threshold=0.3)

    assert len(clusters) == 2
    sizes = sorted(c.member_count for c in clusters)
    assert sizes == [1, 2]


def test_cluster_representative_is_the_highest_confidence_member():
    high = Evidence(document_id="d", page_number=1, chunk_id="high", text="Alpha one", confidence=0.9)
    low = Evidence(document_id="d", page_number=1, chunk_id="low", text="Alpha two", confidence=0.3)

    clusters = cluster_evidence([low, high], embedder=_GroupedEmbedder(), distance_threshold=0.3)

    assert clusters[0].representative.chunk_id == "high"


def test_cluster_type_detects_vision_source():
    clusters = cluster_evidence([_evidence("c1", "A defect was found", source="vision:Defect")])

    assert clusters[0].cluster_type == "vision"


def test_cluster_type_detects_incident_keyword():
    clusters = cluster_evidence([_evidence("c1", "P-101 FAILED_DUE_TO bearing wear", source="graph")])

    assert clusters[0].cluster_type == "incident"


def test_cluster_type_detects_maintenance_keyword():
    clusters = cluster_evidence([_evidence("c1", "P-101 INSPECTED_BY technician", source="graph")])

    assert clusters[0].cluster_type == "maintenance"


def test_cluster_type_falls_back_to_equipment_when_entities_present():
    clusters = cluster_evidence([_evidence("c1", "Some general text about equipment", entity_ids=["e1"])])

    assert clusters[0].cluster_type == "equipment"


def test_cluster_type_falls_back_to_text_with_no_other_signal():
    clusters = cluster_evidence([_evidence("c1", "Some completely generic text")])

    assert clusters[0].cluster_type == "text"


def test_cluster_confidence_is_the_average_of_member_confidences():
    a = Evidence(document_id="d", page_number=1, chunk_id="a", text="Alpha one", confidence=0.8)
    b = Evidence(document_id="d", page_number=1, chunk_id="b", text="Alpha two", confidence=0.4)

    clusters = cluster_evidence([a, b], embedder=_GroupedEmbedder(), distance_threshold=0.3)

    assert clusters[0].confidence == 0.6
