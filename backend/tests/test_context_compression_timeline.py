from app.context_compression.timeline import compress_timeline
from app.retrieval.evidence import Evidence


def _graph_evidence(chunk_id, text, entity_ids):
    return Evidence(document_id="", page_number=0, chunk_id=chunk_id, text=text, entity_ids=entity_ids, source="graph", confidence=0.5)


def _fusion_evidence(chunk_id, text):
    return Evidence(document_id="d", page_number=1, chunk_id=chunk_id, text=text, source="fusion", confidence=0.7)


def test_empty_list_returns_empty():
    assert compress_timeline([]) == []


def test_single_item_passes_through_unchanged():
    item = _graph_evidence("c1", "P-101 INSPECTED_BY tech", ["e1"])

    result = compress_timeline([item])

    assert result == [item]


def test_merges_adjacent_graph_evidence_sharing_an_entity():
    a = _graph_evidence("c1", "P-101 INSPECTED_BY tech", ["e1"])
    b = _graph_evidence("c2", "P-101 MAINTAINED_BY crew", ["e1"])

    result = compress_timeline([a, b])

    assert len(result) == 1
    assert result[0].chunk_id == "episode:c1+c2"
    assert "INSPECTED_BY" in result[0].text and "MAINTAINED_BY" in result[0].text
    assert result[0].entity_ids == ["e1"]


def test_does_not_merge_non_adjacent_evidence_about_the_same_entity():
    a = _graph_evidence("c1", "P-101 INSPECTED_BY tech", ["e1"])
    unrelated = _fusion_evidence("c2", "unrelated text")
    b = _graph_evidence("c3", "P-101 MAINTAINED_BY crew", ["e1"])

    result = compress_timeline([a, unrelated, b])

    assert len(result) == 3  # the unrelated item in between breaks adjacency


def test_does_not_merge_graph_evidence_about_different_entities():
    a = _graph_evidence("c1", "P-101 INSPECTED_BY tech", ["e1"])
    b = _graph_evidence("c2", "V-202 INSPECTED_BY tech", ["e2"])

    result = compress_timeline([a, b])

    assert len(result) == 2


def test_does_not_merge_non_graph_evidence():
    a = _fusion_evidence("c1", "text one")
    b = _fusion_evidence("c2", "text two")

    result = compress_timeline([a, b])

    assert len(result) == 2


def test_merged_confidence_is_the_average_of_members():
    a = Evidence(document_id="", page_number=0, chunk_id="c1", text="P-101 INSPECTED_BY tech", entity_ids=["e1"], source="graph", confidence=0.5)
    b = Evidence(document_id="", page_number=0, chunk_id="c2", text="P-101 MAINTAINED_BY crew", entity_ids=["e1"], source="graph", confidence=0.9)

    result = compress_timeline([a, b])

    assert result[0].confidence == 0.7


def test_merge_preserves_provenance_via_combined_chunk_id():
    a = _graph_evidence("c1", "P-101 INSPECTED_BY tech", ["e1"])
    b = _graph_evidence("c2", "P-101 MAINTAINED_BY crew", ["e1"])
    c = _graph_evidence("c3", "P-101 FAILED_DUE_TO wear", ["e1"])

    result = compress_timeline([a, b, c])

    assert len(result) == 1
    assert result[0].chunk_id == "episode:c1+c2+c3"
