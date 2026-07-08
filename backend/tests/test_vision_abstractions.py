from app.vision.defect.detector import ClassicalCVDefectDetector
from app.vision.detection.detector import ClassicalCVObjectDetector
from app.vision.diagram.parser import ClassicalCVDiagramParser
from app.vision.gauge.reader import GaugeReader
from app.vision.ocr.provider import UnavailableOCRProvider


def test_every_concrete_vision_adapter_satisfies_the_vision_model_contract():
    adapters = [
        UnavailableOCRProvider(),
        ClassicalCVDiagramParser(),
        ClassicalCVObjectDetector(),
        ClassicalCVDefectDetector(),
        GaugeReader(),
    ]
    for adapter in adapters:
        assert isinstance(adapter.name, str) and adapter.name
        assert isinstance(adapter.version, str) and adapter.version
