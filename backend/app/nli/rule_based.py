"""Deterministic NLI fallback — used when no transformer NLI model is
available (see factory.py), and directly testable/auditable on its own.

Reuses the exact same antonym vocabulary `ContradictionDetector` already
uses (app.contradiction.detector) rather than maintaining a second, drifting
list: CONTRADICTION is only claimed when premise and hypothesis contain
opposite terms from that vocabulary. ENTAILMENT is only claimed for very
close paraphrase (high word overlap) — this is a coarse lexical heuristic,
not real inference, so it is conservative by design: most pairs land in
NEUTRAL rather than a confident guess.
"""

import re

from app.contradiction.detector import _STATE_ANTONYM_PAIRS
from app.nli.models import NLILabel, NLIResult

_WORD_PATTERN = re.compile(r"[a-z0-9]+")
_ENTAILMENT_OVERLAP_THRESHOLD = 0.8
_CONTRADICTION_SCORE = 0.75
_ENTAILMENT_SCORE = 0.6
_NEUTRAL_SCORE = 0.5


def _words(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.lower()))


class RuleBasedNLIDetector:
    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()

        for term_x, term_y in _STATE_ANTONYM_PAIRS:
            if (term_x in premise_lower and term_y in hypothesis_lower) or (
                term_y in premise_lower and term_x in hypothesis_lower
            ):
                return NLIResult(premise, hypothesis, NLILabel.CONTRADICTION, _CONTRADICTION_SCORE)

        premise_words, hypothesis_words = _words(premise), _words(hypothesis)
        if premise_words and hypothesis_words:
            overlap = len(premise_words & hypothesis_words) / len(premise_words | hypothesis_words)
            if overlap >= _ENTAILMENT_OVERLAP_THRESHOLD:
                return NLIResult(premise, hypothesis, NLILabel.ENTAILMENT, _ENTAILMENT_SCORE)

        return NLIResult(premise, hypothesis, NLILabel.NEUTRAL, _NEUTRAL_SCORE)
