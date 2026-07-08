"""Equipment Recognition (M7 Component 4).

ClassicalCVObjectDetector is a region-proposal baseline, not equipment-type
classification: it answers "where is the salient object in this photo," via
foreground/background segmentation (Otsu threshold + external contours),
not "what specific pump model is this." Only the largest detected region is
tagged with the caller-supplied `equipment_hint` (if any); every other region
is `equipment_type="unknown"`. A real YOLO/GroundingDINO/Detectron2 model —
this class's documented upgrade path — would replace this with genuine
multi-class classification behind the same `ObjectDetector` Protocol, so
`VisionKnowledgeGraphBuilder` and the API layer never change.
"""

from typing import Protocol

from app.vision.detection.models import Detection
from app.vision.models import BoundingBox

_MIN_REGION_AREA_FRACTION = 0.01  # ignore contours smaller than 1% of the frame (noise)
_MAX_REGION_AREA_FRACTION = 0.95  # ignore a contour that's essentially the whole frame (background)


class ObjectDetector(Protocol):
    def detect(self, image_path, image_id: str, equipment_hint: str | None = None) -> list[Detection]: ...


class ClassicalCVObjectDetector:
    name = "classical_cv_object_detector"
    version = "1.0"

    def detect(self, image_path, image_id: str, equipment_hint: str | None = None) -> list[Detection]:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return []
        frame_area = image.shape[0] * image.shape[1]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_fraction = area / frame_area if frame_area else 0
            if area_fraction < _MIN_REGION_AREA_FRACTION or area_fraction > _MAX_REGION_AREA_FRACTION:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((area_fraction, BoundingBox(x, y, x + w, y + h)))

        candidates.sort(key=lambda c: c[0], reverse=True)

        detections = []
        for i, (area_fraction, bbox) in enumerate(candidates):
            equipment_type = equipment_hint if (i == 0 and equipment_hint) else "unknown"
            detections.append(
                Detection(
                    equipment_type=equipment_type,
                    bounding_box=bbox,
                    confidence=round(min(0.9, 0.3 + area_fraction), 4),
                    image_reference=image_id,
                )
            )
        return detections
