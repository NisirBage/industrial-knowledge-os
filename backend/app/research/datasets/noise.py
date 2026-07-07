"""Noise injection for synthetic documents (M5 Component 4).

Real industrial documents are messy — scanned with OCR errors, missing
fields, contradictory across sources, duplicated, and drifting across
revisions. A benchmark that only ever sees pristine text overstates every
subsystem's real-world accuracy, so NoiseInjector deliberately degrades
generated documents in the same ways real ones are degraded, using the same
seeded RNG the generator uses for full-pipeline determinism.
"""

import random

from app.research.datasets.models import NoiseConfig, SyntheticDocument

_OCR_CONFUSIONS = {"O": "0", "0": "O", "l": "1", "1": "l", "S": "5", "5": "S", "B": "8", "8": "B"}
_STATE_ANTONYM_PAIRS = [
    ("open", "closed"),
    ("running", "stopped"),
    ("pass", "fail"),
    ("normal", "abnormal"),
    ("functional", "failed"),
]


class NoiseInjector:
    def apply(self, document: SyntheticDocument, config: NoiseConfig, rng: random.Random) -> SyntheticDocument:
        text = document.text
        if config.ocr_error_rate > 0:
            text = self._apply_ocr_errors(text, config.ocr_error_rate, rng)
        if config.typo_rate > 0:
            text = self._apply_typos(text, config.typo_rate, rng)
        if config.missing_field_rate > 0 and rng.random() < config.missing_field_rate:
            text = self._drop_a_sentence(text, rng)
        if config.version_drift:
            text = self._apply_version_drift(text, rng)

        metadata = dict(document.metadata)
        metadata["noise_applied"] = "true"
        return SyntheticDocument(
            doc_id=document.doc_id,
            doc_type=document.doc_type,
            text=text,
            ground_truth_entities=document.ground_truth_entities,
            ground_truth_relations=document.ground_truth_relations,
            metadata=metadata,
        )

    def duplicate(self, document: SyntheticDocument, rng: random.Random) -> SyntheticDocument:
        new_id = f"{document.doc_id}-dup-{rng.randint(1000, 9999)}"
        metadata = dict(document.metadata)
        metadata["duplicate_of"] = document.doc_id
        return SyntheticDocument(
            doc_id=new_id,
            doc_type=document.doc_type,
            text=document.text,
            ground_truth_entities=document.ground_truth_entities,
            ground_truth_relations=document.ground_truth_relations,
            metadata=metadata,
        )

    def generate_conflicting(self, document: SyntheticDocument, rng: random.Random) -> SyntheticDocument:
        """Produce a contradicting counterpart: same equipment references,
        opposite state word — the exact pattern ContradictionDetector (M3)
        is designed to catch, so this is the generator that stress-tests it.
        """
        text = document.text
        for term_a, term_b in rng.sample(_STATE_ANTONYM_PAIRS, len(_STATE_ANTONYM_PAIRS)):
            if term_a in text.lower():
                text = text.replace(term_a, term_b).replace(term_a.capitalize(), term_b.capitalize())
                break
            if term_b in text.lower():
                text = text.replace(term_b, term_a).replace(term_b.capitalize(), term_a.capitalize())
                break
        else:
            text = text + " Note: state reported as closed."

        new_id = f"{document.doc_id}-conflict-{rng.randint(1000, 9999)}"
        metadata = dict(document.metadata)
        metadata["conflicts_with"] = document.doc_id
        return SyntheticDocument(
            doc_id=new_id,
            doc_type=document.doc_type,
            text=text,
            ground_truth_entities=document.ground_truth_entities,
            ground_truth_relations=document.ground_truth_relations,
            metadata=metadata,
        )

    @staticmethod
    def _apply_ocr_errors(text: str, error_rate: float, rng: random.Random) -> str:
        chars = list(text)
        for i, char in enumerate(chars):
            if char in _OCR_CONFUSIONS and rng.random() < error_rate:
                chars[i] = _OCR_CONFUSIONS[char]
        return "".join(chars)

    @staticmethod
    def _apply_typos(text: str, typo_rate: float, rng: random.Random) -> str:
        words = text.split(" ")
        for i, word in enumerate(words):
            if len(word) > 3 and rng.random() < typo_rate:
                position = rng.randint(1, len(word) - 2)
                chars = list(word)
                chars[position], chars[position - 1] = chars[position - 1], chars[position]
                words[i] = "".join(chars)
        return " ".join(words)

    @staticmethod
    def _drop_a_sentence(text: str, rng: random.Random) -> str:
        sentences = [s for s in text.split(". ") if s.strip()]
        if len(sentences) <= 1:
            return text
        index_to_drop = rng.randint(0, len(sentences) - 1)
        remaining = sentences[:index_to_drop] + sentences[index_to_drop + 1 :]
        return ". ".join(remaining)

    @staticmethod
    def _apply_version_drift(text: str, rng: random.Random) -> str:
        revision = rng.randint(1, 5)
        return text + f"\n\n[Rev. {revision} — supersedes prior revisions]"
