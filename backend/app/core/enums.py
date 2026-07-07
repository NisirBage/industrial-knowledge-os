"""Shared severity/priority vocabularies used across contradiction detection
and recommendations, so the two components agree on what "high severity"
means instead of each defining their own scale.
"""

from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
