from app.retrieval.evidence import Evidence
from app.retrieval.explain import explain_evidence


def test_explains_fusion_evidence_with_every_real_signal():
    evidence = Evidence(
        document_id="doc-1",
        page_number=1,
        chunk_id="c1",
        text="text",
        bm25_score=3.5,
        vector_score=0.9,
        graph_score=0.15,
        reranker_score=2.1,
        ltr_score=2.1,
        source="fusion",
        confidence=0.85,
    )

    explanation = explain_evidence(evidence, final_rank=1)

    assert "Rank #1" in explanation
    assert "BM25=3.500" in explanation
    assert "vector=0.900" in explanation
    assert "graph_boost=0.150" in explanation
    assert "Cross-encoder score=2.100" in explanation
    assert "Confidence=85%" in explanation


def test_explains_graph_evidence_without_fusion_signals():
    evidence = Evidence(document_id="", page_number=0, chunk_id="graph:1:2", text="P-101 CONNECTED_TO V-202", graph_score=0.55, source="graph", confidence=0.55)

    explanation = explain_evidence(evidence, final_rank=3)

    assert "Rank #3" in explanation
    assert "knowledge graph relation" in explanation
    assert "BM25" not in explanation  # graph-synthesized evidence has no lexical/vector signal to report


def test_mentions_the_ltr_score_only_when_it_differs_from_the_reranker_score():
    same = Evidence(document_id="d", page_number=1, chunk_id="c", text="t", reranker_score=1.0, ltr_score=1.0, source="fusion", confidence=0.7)
    different = Evidence(document_id="d", page_number=1, chunk_id="c", text="t", reranker_score=1.0, ltr_score=0.03, source="fusion", confidence=0.7)

    assert "Learning-to-rank score" not in explain_evidence(same, final_rank=1)
    assert "Learning-to-rank score=0.030" in explain_evidence(different, final_rank=1)
