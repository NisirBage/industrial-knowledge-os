"""Engineering Diagram Understanding (M7 Component 3).

ClassicalCVDiagramParser detects P&ID-style symbols via classical computer
vision, not a trained symbol classifier: Hough circles for valve/instrument
dials, quadrilateral contours for equipment blocks, and Hough line segments
for pipes. This is deliberately simple and explainable, not a fake capability
— a real deep-learning diagram symbol classifier (trained on a labeled P&ID
symbol dataset) is the documented upgrade path, same "extension point, not a
faked capability" pattern M6 used for SHAP. `DiagramParser` is a Protocol so
that upgrade is a new class, not a rewrite of anything that calls this one.

Known limitation, stated plainly: classical shape detection cannot
distinguish a "pump" rectangle from a generic equipment-block rectangle, or a
"valve" circle from a "gauge" circle — every rectangle is labeled PUMP and
every circle is labeled VALVE here. This is coarse but real detection, not
random noise (see docs/architecture/diagram-parser.md for the honest
precision/recall discussion), and every symbol carries its raw detection
confidence and bounding box so a human reviewer (or the Vision Workspace) can
correct the coarse label without losing the underlying detection.
"""

import numpy as np

from app.ontology.labels import EntityLabel
from app.vision.diagram.models import DiagramSymbol, SymbolType
from app.vision.models import BoundingBox

_MIN_EQUIPMENT_CONTOUR_AREA = 200
_MIN_PIPE_LINE_LENGTH = 40


class DiagramParser:
    """Protocol-shaped by convention (duck-typed, like every other
    Provider/Extractor/Detector in this codebase): any class implementing
    `parse(image_path, image_id) -> list[DiagramSymbol]` is a valid parser."""

    def parse(self, image_path, image_id: str) -> list[DiagramSymbol]: ...


class ClassicalCVDiagramParser:
    name = "classical_cv_diagram_parser"
    version = "1.0"

    def parse(self, image_path, image_id: str) -> list[DiagramSymbol]:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        symbols: list[DiagramSymbol] = []
        symbols += self._detect_circles(gray, image_id)
        symbols += self._detect_equipment_blocks(edges, image_id)
        symbols += self._detect_pipes(edges, image_id)
        return symbols

    @staticmethod
    def _detect_circles(gray, image_id: str) -> list[DiagramSymbol]:
        import cv2

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=8, maxRadius=80
        )
        if circles is None:
            return []
        symbols = []
        for x, y, r in np.round(circles[0]).astype(int):
            symbols.append(
                DiagramSymbol(
                    symbol_type=SymbolType.VALVE,
                    bounding_box=BoundingBox(int(x - r), int(y - r), int(x + r), int(y + r)),
                    confidence=0.6,
                    image_id=image_id,
                )
            )
        return symbols

    @staticmethod
    def _detect_equipment_blocks(edges, image_id: str) -> list[DiagramSymbol]:
        import cv2

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        symbols = []
        for contour in contours:
            if cv2.contourArea(contour) < _MIN_EQUIPMENT_CONTOUR_AREA:
                continue
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) != 4:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            symbols.append(
                DiagramSymbol(
                    symbol_type=SymbolType.PUMP,
                    bounding_box=BoundingBox(x, y, x + w, y + h),
                    confidence=0.5,
                    image_id=image_id,
                )
            )
        return symbols

    @staticmethod
    def _detect_pipes(edges, image_id: str) -> list[DiagramSymbol]:
        import cv2

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=_MIN_PIPE_LINE_LENGTH, maxLineGap=5)
        if lines is None:
            return []
        symbols = []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            symbols.append(
                DiagramSymbol(
                    symbol_type=SymbolType.PIPE,
                    bounding_box=BoundingBox(int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))),
                    confidence=0.5,
                    image_id=image_id,
                )
            )
        return symbols


_SYMBOL_TO_ONTOLOGY_LABEL: dict[SymbolType, EntityLabel] = {
    SymbolType.VALVE: EntityLabel.VALVE,
    SymbolType.PUMP: EntityLabel.PUMP,
    SymbolType.PIPE: EntityLabel.PIPE,
    SymbolType.SENSOR: EntityLabel.SENSOR,
    SymbolType.INSTRUMENT: EntityLabel.INSTRUMENT,
}


def ontology_label_for_symbol(symbol_type: SymbolType) -> EntityLabel | None:
    """Maps a detected diagram symbol to the ontology entity it represents —
    FLOW_ARROW and LABEL have no equipment-class ontology counterpart, so
    they map to None (they still become DiagramSymbol graph nodes, just not
    ASSOCIATED_WITH a domain entity)."""
    return _SYMBOL_TO_ONTOLOGY_LABEL.get(symbol_type)
