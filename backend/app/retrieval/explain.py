"""Retrieval Explainability (O1 Component 5): a genuinely computed
human-readable "why was this retrieved" string for one piece of Evidence —
built entirely from real numbers already on `Evidence` (bm25_score,
vector_score, graph_score, reranker_score, ltr_score, confidence), never a
fabricated narrative. `final_rank` is the evidence's 1-indexed position in
the list it was returned in — evidence itself doesn't know this (a frozen
value object has no notion of "which list am I in"), so callers pass it in
at the point they already have the full ordered list.
"""

from app.retrieval.evidence import Evidence


def explain_evidence(evidence: Evidence, final_rank: int) -> str:
    parts = [f"Rank #{final_rank}."]

    if evidence.source == "graph":
        parts.append(f"Synthesized directly from a knowledge graph relation (graph_score={evidence.graph_score:.2f}).")
    else:
        parts.append(f"BM25={evidence.bm25_score:.3f}, vector={evidence.vector_score:.3f}, graph_boost={evidence.graph_score:.3f}.")
        parts.append(f"Cross-encoder score={evidence.reranker_score:.3f}.")
        if evidence.ltr_score != evidence.reranker_score:
            parts.append(f"Learning-to-rank score={evidence.ltr_score:.3f} (this is what determined final ordering).")

    parts.append(f"Confidence={evidence.confidence:.0%}.")
    return " ".join(parts)
