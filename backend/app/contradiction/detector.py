"""Contradiction Detection (M3 Component 5).

Industrial documentation frequently disagrees about equipment state across
sources (an inspection log vs. a maintenance note vs. a sensor reading).
This is a deterministic heuristic, not an NLI model: it looks for evidence
chunks that share a mention of the same equipment tag and assert opposite
states from a small known-antonym vocabulary (open/closed, running/stopped,
pass/fail, ...). It will miss contradictions phrased outside that
vocabulary — the extension point for a learned contradiction/NLI model is
this class's `detect` method, same input/output contract.
"""

import re

from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle

_TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b")

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
        text_items = bundle.evidence
        tags_by_chunk = {item.chunk_id: set(_TAG_PATTERN.findall(item.text)) for item in text_items}

        contradictions: list[Contradiction] = []
        seen_keys: set[tuple[str, str, str]] = set()

        # A single chunk can itself contain both sides of a conflict (e.g. one
        # document quoting an inspection log and a maintenance note that
        # disagree) — chunking doesn't guarantee conflicting statements land
        # in different chunks, so check each chunk against itself first.
        for item in text_items:
            conflict = self._find_conflict(item.text, item.text)
            if conflict is None:
                continue
            term_a, term_b = conflict
            for tag in tags_by_chunk[item.chunk_id]:
                key = (item.chunk_id, item.chunk_id, tag)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                contradictions.append(
                    Contradiction(
                        conflicting_evidence=[item.chunk_id],
                        confidence=item.confidence,
                        severity=self._severity_for(item.confidence),
                        resolution_strategy="Manually verify current state via field inspection or updated records.",
                        recommended_clarification=(
                            f"Evidence conflicts on {tag} within the same source: it states both "
                            f"'{term_a}' and '{term_b}'. Confirm current state before acting."
                        ),
                        description=f"Conflicting state reported for {tag}",
                    )
                )

        for i, item_a in enumerate(text_items):
            for item_b in text_items[i + 1 :]:
                shared_tags = tags_by_chunk[item_a.chunk_id] & tags_by_chunk[item_b.chunk_id]
                if not shared_tags:
                    continue

                conflict = self._find_conflict(item_a.text, item_b.text)
                if conflict is None:
                    continue
                term_a, term_b = conflict

                for tag in shared_tags:
                    key = tuple(sorted([item_a.chunk_id, item_b.chunk_id])) + (tag,)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    confidence = min(item_a.confidence, item_b.confidence)
                    contradictions.append(
                        Contradiction(
                            conflicting_evidence=[item_a.chunk_id, item_b.chunk_id],
                            confidence=confidence,
                            severity=self._severity_for(confidence),
                            resolution_strategy="Manually verify current state via field inspection or updated records.",
                            recommended_clarification=(
                                f"Evidence conflicts on {tag}: one source says '{term_a}', "
                                f"another says '{term_b}'. Confirm current state before acting."
                            ),
                            description=f"Conflicting state reported for {tag}",
                        )
                    )

        return contradictions

    @staticmethod
    def _find_conflict(text_a: str, text_b: str) -> tuple[str, str] | None:
        lowered_a, lowered_b = text_a.lower(), text_b.lower()
        for term_x, term_y in _STATE_ANTONYM_PAIRS:
            if term_x in lowered_a and term_y in lowered_b:
                return term_x, term_y
            if term_y in lowered_a and term_x in lowered_b:
                return term_y, term_x
        return None

    @staticmethod
    def _severity_for(confidence: float) -> Severity:
        if confidence >= _HIGH_SEVERITY_THRESHOLD:
            return Severity.HIGH
        if confidence >= _MEDIUM_SEVERITY_THRESHOLD:
            return Severity.MEDIUM
        return Severity.LOW
