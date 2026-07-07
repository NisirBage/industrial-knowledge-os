"""Lightweight, keyword-based query intent classification.

Deliberately not an LLM call: retrieval needs to stay fast and independent of
the LLM provider (which may be the mock/offline fallback). This is an
extension point — richer intent detection (embeddings-based or LLM-assisted)
can replace `classify_intent` without changing its signature or call sites.
"""

from enum import Enum

_KEYWORDS: dict[str, list[str]] = {
    "maintenance": ["maintenance", "lubricat", "repair", "replace", "schedule", "downtime"],
    "incident": ["incident", "failure", "failed", "accident", "root cause", "rca"],
    "compliance": ["standard", "regulation", "compliance", "iso", "osha", "oisd", "permit", "audit"],
    "inspection": ["inspect", "inspection", "audit finding"],
}


class QueryIntent(str, Enum):
    MAINTENANCE = "maintenance"
    INCIDENT = "incident"
    COMPLIANCE = "compliance"
    INSPECTION = "inspection"
    GENERAL = "general"


def classify_intent(query: str) -> QueryIntent:
    lowered = query.lower()
    for intent_value, keywords in _KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return QueryIntent(intent_value)
    return QueryIntent.GENERAL
