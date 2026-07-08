from pydantic import BaseModel

from app.schemas.chat import EvidenceOut


class VisionDocumentOut(BaseModel):
    id: str
    filename: str
    source_type: str
    content_type: str
    width: int
    height: int
    size_bytes: int
    equipment_hint: str | None


class OCRRegionOut(BaseModel):
    text: str
    confidence: float
    bounding_box: list[int]
    graph_node_id: str | None = None


class DiagramSymbolOut(BaseModel):
    symbol_type: str
    confidence: float
    bounding_box: list[int]
    graph_node_id: str | None = None


class DetectionOut(BaseModel):
    equipment_type: str
    confidence: float
    bounding_box: list[int]
    graph_node_id: str | None = None


class GaugeReadingOut(BaseModel):
    needle_angle_degrees: float | None
    normalized_value: float | None
    confidence: float
    valid: bool
    validation_note: str
    bounding_box: list[int]
    graph_node_id: str | None = None


class DefectOut(BaseModel):
    defect_type: str
    confidence: float
    severity: str
    bounding_box: list[int]
    graph_node_id: str | None = None


class VisionAnalysisOut(BaseModel):
    image_id: str
    ocr_regions: list[OCRRegionOut]
    diagram_symbols: list[DiagramSymbolOut]
    detections: list[DetectionOut]
    gauge_readings: list[GaugeReadingOut]
    defects: list[DefectOut]


class MultimodalQueryRequest(BaseModel):
    query: str
    top_k: int = 8


class MultimodalQueryResponse(BaseModel):
    evidence: list[EvidenceOut]
