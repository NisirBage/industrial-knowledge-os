"""The two vision ML abstractions that cut across every task-specific one
(OCRProvider, DiagramParser, ObjectDetector, DefectDetector, GaugeReader —
each defined in its own submodule next to its concrete implementation).

VisionModel is the common identity contract every concrete adapter below
also satisfies (structurally — Protocols aren't inherited from), so the
Vision Knowledge Graph, benchmarks, or a future model-registry-style tool
can record *which* model configuration produced a given result, mirroring
how SqlModelRegistry already versions every trained ML model (M6). It adds
no new behavior, just a name/version contract.

VisionFeatureExtractor is architecture-only for M7: no concrete
implementation exists. It's the seam a future embedding model (e.g. CLIP or
a fine-tuned ResNet) would fill to produce a fixed-length feature vector for
image-similarity search or as input to a classifier trained on real labeled
data — the same "documented extension point, not a faked capability"
pattern the Explainability Layer's SHAP note (M6) and the Defect Detector's
five architecture-only defect types (M7) both use.
"""

from typing import Protocol


class VisionModel(Protocol):
    name: str
    version: str


class VisionFeatureExtractor(Protocol):
    def extract_features(self, image_path) -> list[float]: ...
