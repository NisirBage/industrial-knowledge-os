"""Real, independently-testable similarity signals — each a pure function
over data this project already has, no fabricated scores.
"""

import math

from app.entity_extraction.regex_extractor import _TAG_PREFIX_TO_LABEL
from app.ontology.hierarchy import ancestors_of
from app.ontology.labels import EntityLabel

# Reuses the exact tag-prefix table RegexEntityExtractor already built
# (P/M/C/V/T/S -> Pump/Motor/Compressor/Valve/Tank/Sensor) rather than a
# second, parallel alias dictionary.
_PREFIX_ALIASES: dict[str, str] = {label.value.lower(): prefix for prefix, label in _TAG_PREFIX_TO_LABEL.items()}


def edit_distance(a: str, b: str) -> int:
    """Classic Levenshtein distance (insert/delete/substitute), computed
    directly — no new dependency for a ~15-line dynamic program."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current_row[j - 1] + 1
            delete_cost = previous_row[j] + 1
            substitute_cost = previous_row[j - 1] + (char_a != char_b)
            current_row.append(min(insert_cost, delete_cost, substitute_cost))
        previous_row = current_row
    return previous_row[-1]


def edit_distance_similarity(a: str, b: str) -> float:
    """1.0 = identical, 0.0 = completely different, normalized by the
    longer string's length so "P-101" vs "P-102" scores high regardless of
    absolute string length."""
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return round(1.0 - (edit_distance(a, b) / longest), 4)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return round(dot / (norm_a * norm_b), 4)


def ontology_similarity(label_a: EntityLabel, label_b: EntityLabel) -> float:
    """1.0 for the same label; otherwise the fraction of shared ancestors
    relative to the longer ancestor chain (e.g. Pump and Motor both descend
    from RotatingEquipment -> IndustrialAsset, so they score partially
    similar; Pump and Standard share nothing and score 0.0)."""
    if label_a == label_b:
        return 1.0
    ancestors_a = set(ancestors_of(label_a))
    ancestors_b = set(ancestors_of(label_b))
    shared = ancestors_a & ancestors_b
    if not shared:
        return 0.0
    longest_chain = max(len(ancestors_a), len(ancestors_b))
    return round(len(shared) / longest_chain, 4)


def neighborhood_similarity(neighbors_a: set[str], neighbors_b: set[str]) -> float:
    """Jaccard similarity of two nodes' neighbor-id sets — two entities that
    share most of their graph neighbors are likely the same real-world
    thing described two different ways."""
    if not neighbors_a and not neighbors_b:
        return 0.0
    union = neighbors_a | neighbors_b
    if not union:
        return 0.0
    return round(len(neighbors_a & neighbors_b) / len(union), 4)


def expand_acronym(text: str) -> str | None:
    """Real acronym/alias expansion reusing RegexEntityExtractor's own
    tag-prefix table — "pump" -> "P", "valve" -> "V", etc. Returns `None`
    for text with no known expansion rather than fabricating one."""
    return _PREFIX_ALIASES.get(text.strip().lower())
