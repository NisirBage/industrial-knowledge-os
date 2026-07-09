from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimilaritySignal:
    name: str
    score: float


@dataclass(frozen=True)
class ResolvedEntity:
    canonical_id: str
    canonical_name: str
    aliases: frozenset[str]
    confidence: float
    merge_reason: str
    signals: tuple[SimilaritySignal, ...] = field(default_factory=tuple)
