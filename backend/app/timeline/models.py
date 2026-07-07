"""Timeline Engine data model.

Distinct from app.evidence.models.TimelineEvent (a per-query, text-date-based
reconstruction built by EvidenceFusionEngine from whatever evidence a single
question happened to retrieve). This TimelineEvent is asset-centric and
graph-structural: it's queryable independent of any question ("show me
P-101's full history") and is built from the durable Knowledge Graph, not
from one query's evidence.
"""

from dataclasses import dataclass
from enum import Enum


class TimelineEventType(str, Enum):
    MAINTENANCE = "maintenance"
    INSPECTION = "inspection"
    INCIDENT = "incident"
    REPLACEMENT = "replacement"
    CALIBRATION = "calibration"
    SHUTDOWN = "shutdown"
    RESTART = "restart"
    PERMIT = "permit"
    RECOMMENDATION = "recommendation"
    SENSOR_EVENT = "sensor_event"


@dataclass(frozen=True)
class TimelineEvent:
    id: str
    equipment: str
    event_type: TimelineEventType
    description: str
    relation_type: str
    # None until ingestion captures temporal metadata on graph relations —
    # ordering still stays deterministic via TimelineEngine's sort key
    # (timestamp first when known, canonical_name as a stable tiebreaker).
    timestamp: str | None = None
