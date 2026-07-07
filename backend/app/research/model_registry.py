"""Model Registry (M5): every experiment references *versioned* components,
so a result can always be traced back to exactly which retriever, extractor,
reasoner, recommender, and prediction model produced it.

One dataclass (ComponentVersion) parametrized by ComponentKind, rather than
six near-identical classes (RetrieverVersion, EntityExtractorVersion, ...) —
they'd have the same three fields and no distinct behavior, so a discriminated
single type is less duplication for the same type-safety. The six names the
spec asks for are provided as typed factory functions instead, which is what
callers actually use.
"""

from dataclasses import dataclass, field
from enum import Enum


class ComponentKind(str, Enum):
    RETRIEVER = "retriever"
    ENTITY_EXTRACTOR = "entity_extractor"
    RELATION_EXTRACTOR = "relation_extractor"
    REASONER = "reasoner"
    RECOMMENDATION = "recommendation"
    PREDICTION = "prediction"


@dataclass(frozen=True)
class ComponentVersion:
    kind: ComponentKind
    name: str
    version: str
    config: dict[str, str] = field(default_factory=dict)


def retriever_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.RETRIEVER, name, version, config)


def entity_extractor_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.ENTITY_EXTRACTOR, name, version, config)


def relation_extractor_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.RELATION_EXTRACTOR, name, version, config)


def reasoner_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.REASONER, name, version, config)


def recommendation_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.RECOMMENDATION, name, version, config)


def prediction_version(name: str, version: str, **config: str) -> ComponentVersion:
    return ComponentVersion(ComponentKind.PREDICTION, name, version, config)
