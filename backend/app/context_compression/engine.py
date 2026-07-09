"""ContextCompressionEngine: orchestrates the full pipeline.

Pipeline order here is dedup-first (exact hash -> near-duplicate removal),
*then* clustering and timeline merge — a reordering of the spec's diagram
(which lists clustering before dedup) purely for efficiency: exact/near
duplicates would end up in the same cluster anyway, so removing them first
just means clustering runs over a smaller, strictly-deduplicated input with
no change in final grouping semantics.

"Summary nodes" are each cluster's real representative Evidence (the
highest-confidence member) — not an LLM-generated abstractive summary. That
would need an LLM call per cluster, working directly against the point of
compression (fewer tokens/lower cost before the LLM call the app already
makes), so it's an explicit, documented scope choice, not an oversight.
"""

from dataclasses import dataclass
from enum import Enum

from app.context_compression.clustering import cluster_evidence
from app.context_compression.dedup import remove_exact_duplicates, remove_near_duplicates
from app.context_compression.models import CompressedEvidenceBundle, CompressionStats
from app.context_compression.timeline import compress_timeline
from app.embeddings.embedder import Embedder
from app.retrieval.evidence import Evidence


class CompressionLevel(str, Enum):
    LIGHT = "light"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True)
class _LevelConfig:
    near_duplicate_threshold: float
    cluster_distance_threshold: float


_LEVEL_CONFIGS: dict[CompressionLevel, _LevelConfig] = {
    CompressionLevel.LIGHT: _LevelConfig(near_duplicate_threshold=0.95, cluster_distance_threshold=0.15),
    CompressionLevel.BALANCED: _LevelConfig(near_duplicate_threshold=0.85, cluster_distance_threshold=0.3),
    CompressionLevel.AGGRESSIVE: _LevelConfig(near_duplicate_threshold=0.75, cluster_distance_threshold=0.5),
}


def _estimate_tokens(evidence: list[Evidence]) -> int:
    """Word-count proxy, not a real tokenizer — documented as an estimate,
    consistent across original/compressed counts so the *ratio* between
    them is meaningful even though the absolute numbers aren't exact
    subword-token counts."""
    return sum(len(e.text.split()) for e in evidence)


class ContextCompressionEngine:
    def __init__(self, embedder: Embedder | None = None):
        self._embedder = embedder

    def compress(self, evidence: list[Evidence], level: CompressionLevel = CompressionLevel.BALANCED) -> CompressedEvidenceBundle:
        config = _LEVEL_CONFIGS[level]
        original_count = len(evidence)
        original_tokens = _estimate_tokens(evidence)

        deduped, exact_duplicate_count = remove_exact_duplicates(evidence)
        deduped, near_duplicate_count = remove_near_duplicates(deduped, threshold=config.near_duplicate_threshold, embedder=self._embedder)

        merged = compress_timeline(deduped)

        clusters = cluster_evidence(merged, embedder=self._embedder, distance_threshold=config.cluster_distance_threshold)
        compressed_evidence = tuple(cluster.representative for cluster in clusters)

        stats = CompressionStats(
            original_count=original_count,
            compressed_count=len(compressed_evidence),
            exact_duplicate_count=exact_duplicate_count,
            near_duplicate_count=near_duplicate_count,
            cluster_count=len(clusters),
            original_tokens_estimate=original_tokens,
            compressed_tokens_estimate=_estimate_tokens(list(compressed_evidence)),
        )

        return CompressedEvidenceBundle(clusters=tuple(clusters), compressed_evidence=compressed_evidence, stats=stats)
