"""Common interface every NLI (Natural Language Inference) implementation
must satisfy — the rule-based fallback and any real transformer model are
interchangeable behind this one method, same pattern as Reranker/Ranker."""

from typing import Protocol

from app.nli.models import NLIResult


class NLIDetector(Protocol):
    def classify(self, premise: str, hypothesis: str) -> NLIResult: ...
