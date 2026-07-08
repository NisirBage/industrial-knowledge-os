"""Industrial ontology: the closed vocabulary of entity and relationship types.

Every entity/relation type used anywhere in the system (extractors, the graph
builder, the graph repository, retrieval) must come from these enums — no
label or relationship name should ever be a hardcoded string elsewhere. This
is what makes the knowledge graph queryable in ontology terms ("find all
RotatingEquipment", not "find things labeled 'PUMP' or 'MOTOR' or ...").
"""

from enum import Enum


class EntityLabel(str, Enum):
    INDUSTRIAL_ASSET = "IndustrialAsset"
    ROTATING_EQUIPMENT = "RotatingEquipment"
    PUMP = "Pump"
    MOTOR = "Motor"
    COMPRESSOR = "Compressor"
    VALVE = "Valve"
    PIPE = "Pipe"
    TANK = "Tank"
    SENSOR = "Sensor"
    INSTRUMENT = "Instrument"

    MAINTENANCE_RECORD = "MaintenanceRecord"
    INSPECTION = "Inspection"
    INCIDENT = "Incident"
    FAILURE_MODE = "FailureMode"
    CORRECTIVE_ACTION = "CorrectiveAction"
    PREVENTIVE_ACTION = "PreventiveAction"

    PERMIT = "Permit"
    PROCEDURE = "Procedure"
    REGULATION = "Regulation"
    STANDARD = "Standard"

    MANUFACTURER = "Manufacturer"
    LOCATION = "Location"
    DEPARTMENT = "Department"
    PERSONNEL = "Personnel"
    DOCUMENT = "Document"

    # Vision Knowledge Graph (M7) — every visual analysis output becomes one
    # of these node types, never an isolated blob outside the graph. See
    # docs/architecture/vision-ontology.md.
    IMAGE = "Image"
    DETECTED_OBJECT = "DetectedObject"
    BOUNDING_BOX = "BoundingBox"
    DEFECT = "Defect"
    DIAGRAM_SYMBOL = "DiagramSymbol"
    GAUGE_READING = "GaugeReading"
    OCR_REGION = "OCRRegion"
    VISUAL_EVIDENCE = "VisualEvidence"


class RelationType(str, Enum):
    CONNECTED_TO = "CONNECTED_TO"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    INSTALLED_IN = "INSTALLED_IN"
    MAINTAINED_BY = "MAINTAINED_BY"
    FAILED_DUE_TO = "FAILED_DUE_TO"
    INSPECTED_BY = "INSPECTED_BY"
    REFERENCES = "REFERENCES"
    GENERATED_FROM = "GENERATED_FROM"
    USES_STANDARD = "USES_STANDARD"
    CAUSES = "CAUSES"
    MITIGATED_BY = "MITIGATED_BY"
    REQUIRES = "REQUIRES"
    RELATED_TO = "RELATED_TO"
    # Document -> Entity: kept distinct from RELATED_TO so "which documents
    # mention X" queries never collide with domain relationships.
    MENTIONS = "MENTIONS"

    # Vision Knowledge Graph (M7)
    DEPICTS = "DEPICTS"  # Image -> the real-world entity it shows
    DETECTED_IN = "DETECTED_IN"  # DetectedObject/Defect/DiagramSymbol -> Image
    LOCATED_AT = "LOCATED_AT"  # DetectedObject -> BoundingBox
    READ_FROM = "READ_FROM"  # GaugeReading/OCRRegion -> Image
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # visual node -> the ontology entity it refers to (e.g. DetectedObject -> Pump)
    OBSERVED_DURING = "OBSERVED_DURING"  # visual node -> Inspection/MaintenanceRecord
    SUPPORTS = "SUPPORTS"  # visual evidence corroborating a textual claim
    CONTRADICTS = "CONTRADICTS"  # visual evidence conflicting with a textual claim
