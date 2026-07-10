"""NLI-augmented Contradiction Detection — a drop-in replacement for
`ContradictionDetector` with the identical `detect(bundle) -> list[Contradiction]`
contract (see app/contradiction/detector.py's own docstring, which names
this exact method as the extension point for a learned contradiction/NLI
model).

Reuses `ContradictionDetector`'s tag-scoped sentence extraction to find
candidate sentence pairs mentioning the same equipment tag, then asks the
injected `NLIDetector` whether the two sentences actually contradict each
other — catching disagreements phrased outside the rule-based detector's
fixed antonym vocabulary, at the cost of one NLI call per same-tag sentence
pair and being only as good as the underlying model (or, with the rule-based
NLI fallback injected, no better than the antonym vocabulary it wraps —
this class's *value* is entirely in what `nli_detector` actually is).
"""

import re
from itertools import combinations

from app.contradiction.models import Contradiction
from app.core.enums import Severity
from app.evidence.models import EvidenceBundle
from app.nli.base import NLIDetector
from app.nli.models import NLILabel

_TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}-\d{2,5}[A-Z]?\b")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

_HIGH_SEVERITY_THRESHOLD = 0.8
_MEDIUM_SEVERITY_THRESHOLD = 0.5


class NLIContradictionDetector:
    def __init__(self, nli_detector: NLIDetector):
        self._nli_detector = nli_detector

    def detect(self, bundle: EvidenceBundle) -> list[Contradiction]:
        tag_sentences: dict[str, list[tuple[str, str, float]]] = {}
        for item in bundle.evidence:
            for sentence in _SENTENCE_SPLIT_PATTERN.split(item.text):
                for tag in _TAG_PATTERN.findall(sentence):
                    tag_sentences.setdefault(tag, []).append((sentence, item.chunk_id, item.confidence))

        contradictions: list[Contradiction] = []
        seen_pairs: set[tuple[str, str, str]] = set()

        for tag, sentences in tag_sentences.items():
            for (sentence_a, chunk_a, conf_a), (sentence_b, chunk_b, conf_b) in combinations(sentences, 2):
                if chunk_a == chunk_b:
                    continue
                pair_key = (tag, *sorted((chunk_a, chunk_b)))
                if pair_key in seen_pairs:
                    continue

                result = self._nli_detector.classify(sentence_a, sentence_b)
                if result.label != NLILabel.CONTRADICTION:
                    continue
                seen_pairs.add(pair_key)

                confidence = round(min(conf_a, conf_b) * result.score, 4)
                contradictions.append(
                    Contradiction(
                        conflicting_evidence=sorted({chunk_a, chunk_b}),
                        confidence=confidence,
                        severity=self._severity_for(confidence),
                        resolution_strategy="Manually verify current state via field inspection or updated records.",
                        recommended_clarification=(
                            f"NLI flagged conflicting statements about {tag}: "
                            f'"{sentence_a.strip()}" vs "{sentence_b.strip()}".'
                        ),
                        description=f"NLI-detected conflicting statement for {tag}",
                        metadata={"detector": "nli"},
                    )
                )

        return contradictions

    @staticmethod
    def _severity_for(confidence: float) -> Severity:
        if confidence >= _HIGH_SEVERITY_THRESHOLD:
            return Severity.HIGH
        if confidence >= _MEDIUM_SEVERITY_THRESHOLD:
            return Severity.MEDIUM
        return Severity.LOW
