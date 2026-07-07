"""Canonical-name normalization used before deduplication.

Equipment-tag-like labels are uppercased (plant tag numbers are conventionally
uppercase, e.g. "p-101" and "P-101" are the same asset); everything else is
whitespace-trimmed only, since over-normalizing free text names (personnel,
locations) would lose meaningful distinctions.

GLiNER and regex frequently disagree on span boundaries for the same
equipment (GLiNER: "Pump P-101", regex: "P-101") — without reconciling that,
they'd dedupe as two different entities and silently fork the knowledge
graph. When a tag-like span contains an embedded tag pattern, the tag itself
becomes the canonical name; the fuller text (e.g. "Pump P-101") is preserved
as an alias by the caller (EntityExtractionService), not lost.
"""

import re

from app.ontology.labels import EntityLabel

_TAG_LIKE_LABELS = {
    EntityLabel.PUMP,
    EntityLabel.MOTOR,
    EntityLabel.COMPRESSOR,
    EntityLabel.VALVE,
    EntityLabel.PIPE,
    EntityLabel.TANK,
    EntityLabel.SENSOR,
    EntityLabel.INSTRUMENT,
    EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.STANDARD,
    EntityLabel.REGULATION,
}

_EMBEDDED_TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b")


def normalize_canonical_name(label: EntityLabel, raw_text: str) -> str:
    text = raw_text.strip()
    if label not in _TAG_LIKE_LABELS:
        return text

    upper = text.upper()
    embedded_tag = _EMBEDDED_TAG_PATTERN.search(upper)
    return embedded_tag.group() if embedded_tag else upper
