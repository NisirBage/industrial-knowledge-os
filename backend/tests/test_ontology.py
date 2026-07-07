from app.ontology.hierarchy import ancestors_of, descendants_of, is_a
from app.ontology.labels import EntityLabel


def test_pump_is_a_rotating_equipment_and_industrial_asset():
    assert is_a(EntityLabel.PUMP, EntityLabel.ROTATING_EQUIPMENT)
    assert is_a(EntityLabel.PUMP, EntityLabel.INDUSTRIAL_ASSET)
    assert not is_a(EntityLabel.PUMP, EntityLabel.MAINTENANCE_RECORD)


def test_ancestors_of_pump_includes_full_chain_to_root():
    chain = ancestors_of(EntityLabel.PUMP)
    assert chain == [EntityLabel.PUMP, EntityLabel.ROTATING_EQUIPMENT, EntityLabel.INDUSTRIAL_ASSET]


def test_ancestors_of_root_label_is_itself():
    assert ancestors_of(EntityLabel.LOCATION) == [EntityLabel.LOCATION]


def test_descendants_of_maintenance_record_includes_incident_and_failure_mode():
    descendants = set(descendants_of(EntityLabel.MAINTENANCE_RECORD))
    assert EntityLabel.INCIDENT in descendants
    assert EntityLabel.FAILURE_MODE in descendants
    assert EntityLabel.INSPECTION in descendants
    assert EntityLabel.PUMP not in descendants


def test_descendants_of_rotating_equipment_includes_pump_motor_compressor():
    descendants = set(descendants_of(EntityLabel.ROTATING_EQUIPMENT))
    assert descendants == {EntityLabel.ROTATING_EQUIPMENT, EntityLabel.PUMP, EntityLabel.MOTOR, EntityLabel.COMPRESSOR}
