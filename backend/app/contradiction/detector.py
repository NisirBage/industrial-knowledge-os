"""Contradiction Detection (M3 Component 5).

Industrial documentation frequently disagrees about equipment state across
sources (an inspection log vs. a maintenance note vs. a sensor reading).
This is a deterministic heuristic, not an NLI model: it looks for evidence
that mentions the same equipment tag alongside opposite states from a small
known-antonym vocabulary (open/closed, running/stopped, pass/fail, ...). It
will miss contradictions phrased outside that vocabulary — the extension
point for a learned contradiction/NLI model is this class's `detect` method,
same input/output contract.

Tag-state association is scoped to the sentence a tag is mentioned in, not
the full (multi-paragraph) evidence chunk — this keeps two unrelated pieces
of equipment discussed in different sentences of the same chunk from being
cross-associated with each other's state. It is still an approximation, not
a dependency parse: a tag mentioned parenthetically inside a sentence about a
*different* asset's state (e.g. "Valve V-2201A (isolation valve for P-101)
was confirmed CLOSED...") will still pick up that state, since sentence-level
co-occurrence is the entire heuristic. Real NLI/coreference resolution is the
honest next step if this proves too coarse in practice.

Findings are deduplicated by (tag, antonym pair) — a real-world disagreement
repeated across many evidence chunks (duplicate ingestion, multiple sources
saying the same thing) produces exactly one Contradiction, with every
supporting chunk id aggregated into it, not one Contradiction per chunk
pair. The previous implementation kept one entry per (chunk_id, chunk_id,
tag) combination, which meant a handful of repeated/duplicate evidence items
could report dozens of "contradictions" for a single real disagreement.
"""

import re
from collections import defaultdict

from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle

_TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

_STATE_ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("open", "closed"),
    ("running", "stopped"),
    ("pass", "fail"),
    ("normal", "abnormal"),
    ("functional", "failed"),
    ("operational", "non-operational"),
]

_HIGH_SEVERITY_THRESHOLD = 0.8
_MEDIUM_SEVERITY_THRESHOLD = 0.5


class ContradictionDetector:
    def detect(self, bundle: EvidenceBundle) -> list[Contradiction]:
        # tag -> term -> [(chunk_id, confidence), ...]
        tag_term_evidence: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

        for item in bundle.evidence:
            for tag, term in self._tag_term_mentions(item.text):
                tag_term_evidence[tag][term].append((item.chunk_id, item.confidence))

        contradictions: list[Contradiction] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for tag, term_evidence in tag_term_evidence.items():
            for term_x, term_y in _STATE_ANTONYM_PAIRS:
                if term_x not in term_evidence or term_y not in term_evidence:
                    continue
                key = (tag, term_x, term_y)
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                combined = term_evidence[term_x] + term_evidence[term_y]
                chunk_ids = sorted({chunk_id for chunk_id, _ in combined})
                confidence = min(conf for _, conf in combined)

                contradictions.append(
                    Contradiction(
                        conflicting_evidence=chunk_ids,
                        confidence=confidence,
                        severity=self._severity_for(confidence),
                        resolution_strategy="Manually verify current state via field inspection or updated records.",
                        recommended_clarification=(
                            f"Evidence conflicts on {tag}: some source(s) say '{term_x}', "
                            f"other(s) say '{term_y}'. Confirm current state before acting."
                        ),
                        description=f"Conflicting state reported for {tag}",
                    )
                )

        return contradictions

    @staticmethod
    def _tag_term_mentions(text: str) -> list[tuple[str, str]]:
        mentions: list[tuple[str, str]] = []
        for sentence in _SENTENCE_SPLIT_PATTERN.split(text):
            tags = _TAG_PATTERN.findall(sentence)
            if not tags:
                continue
            lowered = sentence.lower()
            for term_x, term_y in _STATE_ANTONYM_PAIRS:
                for term in (term_x, term_y):
                    if term in lowered:
                        mentions.extend((tag, term) for tag in tags)
        return mentions

    @staticmethod
    def _severity_for(confidence: float) -> Severity:
        if confidence >= _HIGH_SEVERITY_THRESHOLD:
            return Severity.HIGH
        if confidence >= _MEDIUM_SEVERITY_THRESHOLD:
            return Severity.MEDIUM
        return Severity.LOW
