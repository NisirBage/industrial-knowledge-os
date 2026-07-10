from dataclasses import dataclass
from enum import Enum


class NLILabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class NLIResult:
    premise: str
    hypothesis: str
    label: NLILabel
    score: float  # confidence in `label`, in [0, 1]
