"""Is-a hierarchy over EntityLabel.

The hackathon problem statement's ontology sketch was a flat list without
explicit parent/child pairs; the tree below is our interpretation, grouping
labels under the abstract categories implied by industrial domain modeling
(e.g. Pump/Motor/Compressor are all RotatingEquipment). This is what lets
graph queries reason at the right altitude — "show me all RotatingEquipment
incidents" without enumerating every concrete equipment type by hand.
"""

from app.ontology.labels import EntityLabel

_PARENT: dict[EntityLabel, EntityLabel | None] = {
    EntityLabel.INDUSTRIAL_ASSET: None,
    EntityLabel.ROTATING_EQUIPMENT: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.PUMP: EntityLabel.ROTATING_EQUIPMENT,
    EntityLabel.MOTOR: EntityLabel.ROTATING_EQUIPMENT,
    EntityLabel.COMPRESSOR: EntityLabel.ROTATING_EQUIPMENT,
    EntityLabel.VALVE: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.PIPE: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.TANK: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.SENSOR: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.INSTRUMENT: EntityLabel.INDUSTRIAL_ASSET,
    EntityLabel.MAINTENANCE_RECORD: None,
    EntityLabel.INSPECTION: EntityLabel.MAINTENANCE_RECORD,
    EntityLabel.INCIDENT: EntityLabel.MAINTENANCE_RECORD,
    EntityLabel.FAILURE_MODE: EntityLabel.INCIDENT,
    EntityLabel.CORRECTIVE_ACTION: EntityLabel.MAINTENANCE_RECORD,
    EntityLabel.PREVENTIVE_ACTION: EntityLabel.MAINTENANCE_RECORD,
    EntityLabel.PERMIT: None,
    EntityLabel.PROCEDURE: None,
    EntityLabel.REGULATION: None,
    EntityLabel.STANDARD: EntityLabel.REGULATION,
    EntityLabel.MANUFACTURER: None,
    EntityLabel.LOCATION: None,
    EntityLabel.DEPARTMENT: None,
    EntityLabel.PERSONNEL: None,
    EntityLabel.DOCUMENT: None,
}


def parent_of(label: EntityLabel) -> EntityLabel | None:
    return _PARENT[label]


def ancestors_of(label: EntityLabel) -> list[EntityLabel]:
    """Return [label, parent, grandparent, ...] up to the root."""
    chain = [label]
    current = parent_of(label)
    while current is not None:
        chain.append(current)
        current = parent_of(current)
    return chain


def is_a(label: EntityLabel, ancestor: EntityLabel) -> bool:
    return ancestor in ancestors_of(label)


def descendants_of(label: EntityLabel) -> list[EntityLabel]:
    """Return [label, child, grandchild, ...] — label plus every label that is_a it.

    Used to answer ontology-level graph queries such as "every node that is a
    kind of MaintenanceRecord" without hardcoding the concrete label list at
    each call site.
    """
    result = [label]
    frontier = [label]
    while frontier:
        current = frontier.pop()
        children = [child for child, parent in _PARENT.items() if parent == current]
        result.extend(children)
        frontier.extend(children)
    return result
