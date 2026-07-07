from app.timeline.engine import TimelineEngine
from app.timeline.models import TimelineEventType


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        return [
            {"id": "n-inspection", "type": "Inspection", "canonical_name": "Quarterly Inspection", "relation_type": "INSPECTED_BY"},
            {"id": "n-standard", "type": "Standard", "canonical_name": "ISO 9001", "relation_type": "USES_STANDARD"},
        ]

    def maintenance_history(self, canonical_name: str):
        return [{"id": "n-maint", "type": "CorrectiveAction", "canonical_name": "Bearing Replacement", "relation_type": "MAINTAINED_BY"}]

    def incident_history(self, canonical_name: str):
        return [{"id": "n-incident", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]


def test_equipment_timeline_filters_to_timeline_relevant_types_only():
    engine = TimelineEngine(_FakeGraphRepository())

    events = engine.equipment_timeline("P-101")

    event_types = {e.event_type for e in events}
    assert event_types == {TimelineEventType.INSPECTION}  # Standard is excluded, not a timeline event


def test_equipment_timeline_empty_when_entity_not_found():
    engine = TimelineEngine(_FakeGraphRepository())
    assert engine.equipment_timeline("UNKNOWN") == []


def test_maintenance_timeline_classifies_corrective_action_as_maintenance():
    engine = TimelineEngine(_FakeGraphRepository())
    events = engine.maintenance_timeline("P-101")

    assert len(events) == 1
    assert events[0].event_type == TimelineEventType.MAINTENANCE
    assert events[0].description == "P-101 MAINTAINED_BY Bearing Replacement"


def test_incident_timeline_classifies_incident():
    engine = TimelineEngine(_FakeGraphRepository())
    events = engine.incident_timeline("P-101")

    assert len(events) == 1
    assert events[0].event_type == TimelineEventType.INCIDENT


def test_full_history_deduplicates_and_combines_all_sources():
    engine = TimelineEngine(_FakeGraphRepository())
    events = engine.full_history("P-101")

    ids = {e.id for e in events}
    assert ids == {"n-inspection", "n-maint", "n-incident"}


def test_ordering_is_deterministic_when_timestamps_are_unknown():
    engine = TimelineEngine(_FakeGraphRepository())

    first_run = [e.id for e in engine.full_history("P-101")]
    second_run = [e.id for e in engine.full_history("P-101")]

    assert first_run == second_run
