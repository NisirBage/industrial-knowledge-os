from dataclasses import dataclass


@dataclass(frozen=True)
class NamedMetric:
    name: str
    value: float
