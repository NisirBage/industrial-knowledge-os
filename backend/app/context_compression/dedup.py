"""Evidence Deduplication (O1.2 Component 4).

Two tiers, cheapest first:
1. Exact-duplicate hashing — normalized text hash, catches verbatim repeats
   (e.g. the same chunk surfacing from both a vector hit and a BM25 hit)
   for free, before any embedding call.
2. Near-duplicate detection — a documented average of whichever real
   signals apply: text word-overlap (Jaccard, always available), shared
   entity ids (Evidence.entity_ids, always available), and embedding cosine
   similarity (only when an Embedder is supplied — optional, matching the
   graceful-degradation pattern used throughout this project).

Never removes evidence with no duplicate — `deduplicate()` always keeps at
least one representative per group, so unique evidence is never dropped.
"""

import hashlib

from app.embeddings.embedder import Embedder
from app.entity_resolution.similarity import cosine_similarity
from app.retrieval.evidence import Evidence

_DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.85


def normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def text_overlap(a: str, b: str) -> float:
    words_a, words_b = set(a.lower().split()), set(b.lower().split())
    if not words_a and not words_b:
        return 1.0
    union = words_a | words_b
    if not union:
        return 0.0
    return round(len(words_a & words_b) / len(union), 4)


def entity_overlap(a: Evidence, b: Evidence) -> float:
    entities_a, entities_b = set(a.entity_ids), set(b.entity_ids)
    if not entities_a and not entities_b:
        return 0.0
    union = entities_a | entities_b
    if not union:
        return 0.0
    return round(len(entities_a & entities_b) / len(union), 4)


def near_duplicate_score(a: Evidence, b: Evidence, embedder: Embedder | None = None, embedding_cache: dict[str, list[float]] | None = None) -> float:
    signals = [text_overlap(a.text, b.text)]
    if a.entity_ids or b.entity_ids:
        signals.append(entity_overlap(a, b))
    if embedder is not None:
        cache = embedding_cache if embedding_cache is not None else {}
        vector_a = cache.get(a.chunk_id) or embedder.embed_texts([a.text])[0]
        vector_b = cache.get(b.chunk_id) or embedder.embed_texts([b.text])[0]
        cache[a.chunk_id], cache[b.chunk_id] = vector_a, vector_b
        signals.append(cosine_similarity(vector_a, vector_b))
    return round(sum(signals) / len(signals), 4)


def remove_exact_duplicates(evidence: list[Evidence]) -> tuple[list[Evidence], int]:
    """Keeps the first-seen item per normalized-text hash. Returns
    (deduplicated_list, exact_duplicate_count)."""
    seen: dict[str, Evidence] = {}
    for item in evidence:
        key = normalized_text_hash(item.text)
        if key not in seen:
            seen[key] = item
    return list(seen.values()), len(evidence) - len(seen)


def remove_near_duplicates(
    evidence: list[Evidence], threshold: float = _DEFAULT_NEAR_DUPLICATE_THRESHOLD, embedder: Embedder | None = None
) -> tuple[list[Evidence], int]:
    """Greedy: for each item (in order), keep it unless it scores above
    `threshold` against an already-kept item — the highest-confidence item
    in a near-duplicate cluster is not specially preferred here (kept =
    first seen), matching the deterministic, order-preserving behavior the
    rest of this project's retrieval pipeline already has."""
    kept: list[Evidence] = []
    embedding_cache: dict[str, list[float]] = {}
    removed = 0
    for item in evidence:
        is_duplicate = any(near_duplicate_score(item, existing, embedder, embedding_cache) >= threshold for existing in kept)
        if is_duplicate:
            removed += 1
        else:
            kept.append(item)
    return kept, removed
