"""Feature Store data model (M6 Component 2). Every feature the store
serves has a registered FeatureDefinition — the metadata a real feature
store needs to answer "what is this number, where did it come from, and can
I trust it" before it goes anywhere near a trained model.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    description: str
    source: str
    owner: str
    datatype: str
    version: str
    generation_time: str
