"""EvidenceBundle: the single object every stage downstream of retrieval
consumes — never a raw list of strings or dicts. Produced by
EvidenceFusionEngine (fusion.py) from a completed TaskGraph plus
knowledge-graph expansion evidence.
"""

from dataclasses import dataclass, field

from app.contradiction.models import Contradiction
from app.retrieval.evidence import Evidence


@dataclass(frozen=True)
class TimelineEvent:
    description: str
    timestamp: str | None
    source_chunk_id: str
    confidence: float


@dataclass(frozen=True)
class MetadataEvidence:
    key: str
    value: str
    source_chunk_id: str


@dataclass(frozen=True)
class EvidenceBundle:
    evidence: list[Evidence]
    graph_evidence: list[Evidence]
    timeline: list[TimelineEvent]
    metadata: list[MetadataEvidence]
    missing_information: list[str]
    confidence: float
    contradictions: list[Contradiction] = field(default_factory=list)

    def all_evidence(self) -> list[Evidence]:
        return self.evidence + self.graph_evidence
