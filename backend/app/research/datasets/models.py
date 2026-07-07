"""Synthetic dataset data model (M5 Component 4).

Every SyntheticDocument carries its own ground truth (the entities/relations
the generator actually inserted) — the whole point of synthetic generation
here is that labeling is free: we know what's in the text because we wrote
it, so entity/relation-extraction benchmarks always have something correct
to compare against, unlike real industrial documents which need manual
annotation.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SyntheticDocument:
    doc_id: str
    doc_type: str
    text: str
    ground_truth_entities: list[tuple[str, str]]  # (label, canonical_name)
    ground_truth_relations: list[tuple[str, str, str]]  # (source, relation_type, target)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TimelineEventSample:
    equipment_tag: str
    event_type: str
    description: str
    date: str


@dataclass(frozen=True)
class RecommendationOutcomeSample:
    recommendation_description: str
    accepted: bool


@dataclass(frozen=True)
class NoiseConfig:
    ocr_error_rate: float = 0.0
    typo_rate: float = 0.0
    missing_field_rate: float = 0.0
    duplicate_rate: float = 0.0
    conflicting_rate: float = 0.0
    version_drift: bool = False
