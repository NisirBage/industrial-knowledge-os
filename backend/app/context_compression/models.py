from dataclasses import dataclass

from app.retrieval.evidence import Evidence


@dataclass(frozen=True)
class EvidenceCluster:
    cluster_id: str
    cluster_type: str
    representative: Evidence
    members: tuple[Evidence, ...]

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def confidence(self) -> float:
        if not self.members:
            return 0.0
        return round(sum(e.confidence for e in self.members) / len(self.members), 4)


@dataclass(frozen=True)
class CompressionStats:
    original_count: int
    compressed_count: int
    exact_duplicate_count: int
    near_duplicate_count: int
    cluster_count: int
    original_tokens_estimate: int
    compressed_tokens_estimate: int

    @property
    def compression_ratio(self) -> float:
        if self.original_count == 0:
            return 0.0
        return round(self.compressed_count / self.original_count, 4)

    @property
    def token_reduction_ratio(self) -> float:
        if self.original_tokens_estimate == 0:
            return 0.0
        return round(1.0 - (self.compressed_tokens_estimate / self.original_tokens_estimate), 4)

    @property
    def summary(self) -> str:
        return (
            f"Compressed {self.original_count} chunks into {self.compressed_count} evidence groups "
            f"({self.cluster_count} clusters, {self.exact_duplicate_count} exact + {self.near_duplicate_count} near duplicates removed, "
            f"~{self.token_reduction_ratio:.0%} token reduction)."
        )


@dataclass(frozen=True)
class CompressedEvidenceBundle:
    clusters: tuple[EvidenceCluster, ...]
    compressed_evidence: tuple[Evidence, ...]
    stats: CompressionStats
