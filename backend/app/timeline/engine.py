"""Timeline Engine (M4 Component 1): chronological equipment histories built
from the durable Knowledge Graph.

Every method returns events in a deterministic order (timestamp first when
known, then canonical description as a stable tiebreaker) regardless of the
graph backend's own iteration/query order — required so predictions built
from a timeline are reproducible.
"""

from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.timeline.models import TimelineEvent, TimelineEventType

_LABEL_TO_EVENT_TYPE: dict[str, TimelineEventType] = {
    EntityLabel.INSPECTION.value: TimelineEventType.INSPECTION,
    EntityLabel.INCIDENT.value: TimelineEventType.INCIDENT,
    EntityLabel.FAILURE_MODE.value: TimelineEventType.INCIDENT,
    EntityLabel.MAINTENANCE_RECORD.value: TimelineEventType.MAINTENANCE,
    EntityLabel.CORRECTIVE_ACTION.value: TimelineEventType.MAINTENANCE,
    EntityLabel.PREVENTIVE_ACTION.value: TimelineEventType.MAINTENANCE,
    EntityLabel.PERMIT.value: TimelineEventType.PERMIT,
}


class TimelineEngine:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def equipment_timeline(self, canonical_name: str) -> list[TimelineEvent]:
        node = self._graph_repository.find_entity(canonical_name)
        if node is None:
            return []
        neighbors = self._graph_repository.find_neighbors(node["id"], depth=1)
        return self._to_events(canonical_name, neighbors)

    def maintenance_timeline(self, canonical_name: str) -> list[TimelineEvent]:
        return self._to_events(canonical_name, self._graph_repository.maintenance_history(canonical_name))

    def incident_timeline(self, canonical_name: str) -> list[TimelineEvent]:
        return self._to_events(canonical_name, self._graph_repository.incident_history(canonical_name))

    def full_history(self, canonical_name: str) -> list[TimelineEvent]:
        combined: dict[str, TimelineEvent] = {}
        for event in (
            self.equipment_timeline(canonical_name)
            + self.maintenance_timeline(canonical_name)
            + self.incident_timeline(canonical_name)
        ):
            combined[event.id] = event
        return self._sorted(list(combined.values()))

    def _to_events(self, equipment: str, records: list[dict[str, str]]) -> list[TimelineEvent]:
        events: list[TimelineEvent] = []
        for record in records:
            event_type = _LABEL_TO_EVENT_TYPE.get(record["type"])
            if event_type is None:
                continue
            relation_type = record.get("relation_type") or "RELATED_TO"
            events.append(
                TimelineEvent(
                    id=record["id"],
                    equipment=equipment,
                    event_type=event_type,
                    description=f"{equipment} {relation_type} {record['canonical_name']}",
                    relation_type=relation_type,
                )
            )
        return self._sorted(events)

    @staticmethod
    def _sorted(events: list[TimelineEvent]) -> list[TimelineEvent]:
        return sorted(events, key=lambda e: (e.timestamp is None, e.timestamp or "", e.description))
