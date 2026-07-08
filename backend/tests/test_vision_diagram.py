import numpy as np
import pytest

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from app.ontology.labels import EntityLabel
from app.vision.diagram.models import SymbolType
from app.vision.diagram.parser import ClassicalCVDiagramParser, ontology_label_for_symbol

pytestmark = pytest.mark.skipif(not _HAS_CV2, reason="opencv-python-headless not installed")


def _simple_pid_image(path):
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.circle(img, (80, 80), 30, (0, 0, 0), 3)  # valve
    cv2.rectangle(img, (200, 50), (320, 150), (0, 0, 0), 3)  # equipment block
    cv2.line(img, (30, 200), (250, 200), (0, 0, 0), 3)  # pipe
    cv2.imwrite(str(path), img)
    return path


def test_parser_detects_circle_as_valve(tmp_path):
    path = _simple_pid_image(tmp_path / "pid.png")
    parser = ClassicalCVDiagramParser()

    symbols = parser.parse(path, image_id="img-1")

    valves = [s for s in symbols if s.symbol_type == SymbolType.VALVE]
    assert len(valves) >= 1
    assert all(s.image_id == "img-1" for s in symbols)


def test_parser_detects_rectangle_as_equipment_block(tmp_path):
    path = _simple_pid_image(tmp_path / "pid.png")
    parser = ClassicalCVDiagramParser()

    symbols = parser.parse(path, image_id="img-1")

    pumps = [s for s in symbols if s.symbol_type == SymbolType.PUMP]
    assert len(pumps) >= 1


def test_parser_detects_line_as_pipe(tmp_path):
    path = _simple_pid_image(tmp_path / "pid.png")
    parser = ClassicalCVDiagramParser()

    symbols = parser.parse(path, image_id="img-1")

    pipes = [s for s in symbols if s.symbol_type == SymbolType.PIPE]
    assert len(pipes) >= 1


def test_parser_returns_empty_list_for_blank_image(tmp_path):
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), np.full((100, 100, 3), 255, dtype=np.uint8))
    parser = ClassicalCVDiagramParser()

    symbols = parser.parse(path, image_id="img-blank")

    assert symbols == []


def test_parser_returns_empty_list_for_unreadable_path(tmp_path):
    parser = ClassicalCVDiagramParser()

    assert parser.parse(tmp_path / "does_not_exist.png", image_id="img-x") == []


def test_ontology_label_mapping_covers_equipment_symbol_types():
    assert ontology_label_for_symbol(SymbolType.VALVE) == EntityLabel.VALVE
    assert ontology_label_for_symbol(SymbolType.PUMP) == EntityLabel.PUMP
    assert ontology_label_for_symbol(SymbolType.PIPE) == EntityLabel.PIPE
    assert ontology_label_for_symbol(SymbolType.FLOW_ARROW) is None
