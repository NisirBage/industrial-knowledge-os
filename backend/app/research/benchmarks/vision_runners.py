"""Vision benchmark runners (M7 Component 10) — extends the M5 Benchmark
Suite, not a parallel framework: same `BenchmarkReport` shape, same
`ExperimentManager` integration via the `/research/benchmarks/{name}/run`
route, same "measure the real production component" discipline. Every
runner here generates its own small, seeded, deterministic synthetic images
in-memory (matching how `SyntheticDatasetGenerator` produces text documents
in-memory) rather than depending on files on disk.
"""

import math
import tempfile
import time
from pathlib import Path

from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.models import NamedMetric
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import GraphRepositoryProtocol, InMemoryGraphRepository
from app.research.benchmarks.models import BenchmarkReport
from app.retrieval.evidence import Evidence
from app.vision.defect.detector import ClassicalCVDefectDetector
from app.vision.defect.models import Defect, DefectType
from app.vision.detection.detector import ClassicalCVObjectDetector
from app.vision.diagram.parser import ClassicalCVDiagramParser
from app.vision.diagram.models import SymbolType
from app.vision.gauge.reader import GaugeReader
from app.vision.graph.builder import VisionKnowledgeGraphBuilder
from app.vision.ingestion.models import VisionDocument, VisionSourceType
from app.vision.models import BoundingBox
from app.vision.ocr.models import OCRRegion
from app.vision.ocr.provider import OCRProvider
from app.vision.retrieval.multimodal import MultimodalRetriever


def _bench_dir() -> Path:
    directory = Path(tempfile.gettempdir()) / "ikios_vision_benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _rust_patch_array(seed_offset: int = 0):
    import cv2
    import numpy as np

    img = np.full((200, 200, 3), 200, dtype=np.uint8)
    offset = seed_offset * 5
    cv2.rectangle(img, (50 + offset, 50), (150 + offset, 150), (14, 65, 183), -1)
    return img


def _crack_array():
    import cv2
    import numpy as np

    img = np.full((200, 200, 3), 220, dtype=np.uint8)
    cv2.line(img, (10, 100), (190, 108), (0, 0, 0), 1)
    return img


def _gauge_array(needle_angle_degrees: float):
    import cv2
    import numpy as np

    center, radius = (150, 150), 90
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(img, center, radius, (0, 0, 0), 3)
    length = radius * 0.75
    angle_rad = math.radians(needle_angle_degrees)
    dx, dy = length * math.sin(angle_rad), -length * math.cos(angle_rad)
    endpoint = (int(center[0] + dx), int(center[1] + dy))
    cv2.line(img, center, endpoint, (0, 0, 0), 4)
    return img, center, radius


def _pid_array():
    import cv2
    import numpy as np

    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    cv2.circle(img, (80, 80), 30, (0, 0, 0), 3)
    cv2.rectangle(img, (200, 50), (320, 150), (0, 0, 0), 3)
    cv2.line(img, (30, 200), (250, 200), (0, 0, 0), 3)
    return img


def _text_array(text: str):
    import cv2
    import numpy as np

    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)
    return img


class OCRBenchmark:
    """Purpose: measure OCR text-recognition accuracy against rendered
    equipment tags with known ground truth. Limitations: rendered
    typed text, not a real photograph's lighting/angle/wear variation, so
    real-world accuracy will be lower than this suggests."""

    name = "vision_ocr"

    def __init__(self, ocr_provider: OCRProvider, tags: list[str] | None = None):
        self._ocr_provider = ocr_provider
        self._tags = tags or ["P-101", "V-2201A", "M-500", "T-300", "PG-201"]

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        correct = 0
        confidences = []
        for i, tag in enumerate(self._tags):
            path = str(_bench_dir() / f"_ocr_bench_{i}.png")
            cv2.imwrite(path, _text_array(tag))
            regions = self._ocr_provider.extract_text(path, image_id=f"bench-{i}")
            if any(tag.replace("-", "") in r.text.replace("-", "").replace(" ", "") for r in regions):
                correct += 1
            confidences.extend(r.confidence for r in regions)

        accuracy = round(correct / len(self._tags), 4) if self._tags else 0.0
        mean_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        metrics = [NamedMetric("tag_read_accuracy", accuracy), NamedMetric("mean_confidence", mean_confidence)]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_rendered_tags", len(self._tags))


class DiagramParsingBenchmark:
    """Purpose: measure symbol-type recall on a simple P&ID with one known
    valve, one known pump, one known pipe. Limitations: a single fixed
    scene, not a distribution of diagrams — see docs/benchmarks/README.md's
    methodology notes for why single-scene benchmarks understate variance."""

    name = "vision_diagram_parsing"

    def __init__(self, parser: ClassicalCVDiagramParser):
        self._parser = parser

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        path = str(_bench_dir() / "_diagram_bench.png")
        cv2.imwrite(path, _pid_array())
        symbols = self._parser.parse(path, image_id="bench")

        found_types = {s.symbol_type for s in symbols}
        expected_types = {SymbolType.VALVE, SymbolType.PUMP, SymbolType.PIPE}
        recall = round(len(found_types & expected_types) / len(expected_types), 4)
        metrics = [
            NamedMetric("symbol_type_recall", recall),
            NamedMetric("total_symbols_detected", float(len(symbols))),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_pid", 1)


class EquipmentRecognitionBenchmark:
    """Purpose: measure region-proposal recall — does the classical-CV
    baseline find at least one salient region in a photo with one obvious
    foreground object. Limitations: measures localization, not multi-class
    equipment-type classification (out of scope for this baseline)."""

    name = "vision_equipment_recognition"

    def __init__(self, detector: ClassicalCVObjectDetector, num_samples: int = 5):
        self._detector = detector
        self._num_samples = num_samples

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        found_count = 0
        for i in range(self._num_samples):
            path = str(_bench_dir() / f"_equip_bench_{i}.png")
            cv2.imwrite(path, _rust_patch_array(seed_offset=0))
            detections = self._detector.detect(path, image_id=f"bench-{i}")
            if detections:
                found_count += 1

        recall = round(found_count / self._num_samples, 4) if self._num_samples else 0.0
        metrics = [NamedMetric("region_detection_recall", recall)]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_photos", self._num_samples)


class GaugeReadingBenchmark:
    """Purpose: measure gauge-reading accuracy (mean absolute error against
    the exact known needle angle/value) across several needle positions.
    This is a genuine numeric accuracy benchmark, not a pass/fail check."""

    name = "vision_gauge_reading"

    def __init__(self, reader: GaugeReader, angles_degrees: list[float] | None = None):
        self._reader = reader
        self._angles = angles_degrees or [-120.0, -60.0, 0.0, 60.0, 120.0]

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        errors = []
        valid_count = 0
        for i, angle in enumerate(self._angles):
            img, center, radius = _gauge_array(angle)
            path = str(_bench_dir() / f"_gauge_bench_{i}.png")
            cv2.imwrite(path, img)
            readings = self._reader.read_all(path, image_id=f"bench-{i}")
            expected_value = (angle - (-135.0)) / 270.0
            for reading in readings:
                if reading.valid and reading.normalized_value is not None:
                    errors.append(abs(reading.normalized_value - expected_value))
                    valid_count += 1

        mae = round(sum(errors) / len(errors), 4) if errors else None
        metrics = [
            NamedMetric("mean_absolute_error", mae if mae is not None else 1.0),
            NamedMetric("valid_reading_rate", round(valid_count / len(self._angles), 4)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_gauges", len(self._angles))


class DefectDetectionBenchmark:
    """Purpose: measure detection recall for the two implemented defect
    types (corrosion, crack) against images with a known defect present.
    Limitations: only the two classically-detectable defect types are
    measured — the other five are architecture-only (see
    app/vision/defect/detector.py's docstring)."""

    name = "vision_defect_detection"

    def __init__(self, detector: ClassicalCVDefectDetector):
        self._detector = detector

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        corrosion_path, crack_path = str(_bench_dir() / "_defect_bench_corrosion.png"), str(_bench_dir() / "_defect_bench_crack.png")
        cv2.imwrite(corrosion_path, _rust_patch_array())
        cv2.imwrite(crack_path, _crack_array())

        corrosion_found = any(d.defect_type == DefectType.CORROSION for d in self._detector.detect(corrosion_path, "bench-corrosion"))
        crack_found = any(d.defect_type == DefectType.CRACK for d in self._detector.detect(crack_path, "bench-crack"))

        recall = round((int(corrosion_found) + int(crack_found)) / 2, 4)
        metrics = [
            NamedMetric("defect_detection_recall", recall),
            NamedMetric("corrosion_detected", float(corrosion_found)),
            NamedMetric("crack_detected", float(crack_found)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_defect_images", 2)


class _EmptyTextRetriever:
    """Text retrieval quality is already covered by HybridRetrievalBenchmark
    — this stub isolates MultimodalRetrievalBenchmark to measuring exactly
    what Component 8 added: whether visual evidence attached to an entity in
    the graph is actually surfaced for a query naming that entity."""

    def retrieve(self, query: str, top_k: int = 8) -> list[Evidence]:
        return []


class MultimodalRetrievalBenchmark:
    """Purpose: measure visual-evidence retrieval recall — of N equipment
    entities with visual evidence attached in the graph and M with none, does
    querying by name return visual evidence for exactly the ones that have
    it? Builds its own isolated in-memory graph (synthetic ground truth,
    same "own dataset, no live-state pollution" discipline
    EntityExtractionBenchmark's synthetic documents use) rather than writing
    benchmark data into the real production graph. Limitations: measures
    graph traversal correctness, not text-retrieval quality (already covered
    by HybridRetrievalBenchmark) or vision-detector accuracy (covered by the
    other vision benchmarks here).
    """

    name = "vision_multimodal_retrieval"

    def __init__(self, tags_with_evidence: list[str] | None = None, tags_without_evidence: list[str] | None = None):
        # All "P-" tags (-> EntityLabel.PUMP under RegexEntityExtractor's
        # ISA-5.1 prefix table) and all "T-" tags (-> TANK) respectively, so
        # the node label created here always matches the label the query's
        # entity extraction will resolve to.
        self._tags_with_evidence = tags_with_evidence or ["P-101", "P-201", "P-301"]
        self._tags_without_evidence = tags_without_evidence or ["T-300", "T-400"]

    async def run(self) -> BenchmarkReport:
        start = time.perf_counter()
        graph_repository: GraphRepositoryProtocol = InMemoryGraphRepository()
        graph_repository.ensure_constraints()
        builder = VisionKnowledgeGraphBuilder(graph_repository)

        for tag in self._tags_with_evidence:
            graph_repository.create_node(EntityLabel.PUMP, canonical_name=tag, confidence=1.0)
            document = VisionDocument(
                id=f"doc-{tag}", filename=f"{tag}.png", source_type=VisionSourceType.EQUIPMENT_PHOTO,
                content_type="image/png", width=200, height=200, storage_path=f"/tmp/{tag}.png", size_bytes=0, equipment_hint=tag,
            )
            image_node_id = builder.add_image(document)
            builder.add_ocr_regions(image_node_id, [OCRRegion(tag, 0.9, BoundingBox(0, 0, 10, 10), document.id)])
            builder.add_defects(image_node_id, [Defect(DefectType.CORROSION, BoundingBox(0, 0, 20, 20), 0.7, document.id, "medium")])

        for tag in self._tags_without_evidence:
            graph_repository.create_node(EntityLabel.TANK, canonical_name=tag, confidence=1.0)  # matches "T-" -> TANK

        entity_extraction_service = EntityExtractionService(RegexEntityExtractor())
        retriever = MultimodalRetriever(_EmptyTextRetriever(), graph_repository, entity_extraction_service)

        true_positive = sum(1 for tag in self._tags_with_evidence if any(e.source.startswith("vision:") for e in retriever.retrieve(f"What is the condition of {tag}?")))
        false_positive = sum(1 for tag in self._tags_without_evidence if any(e.source.startswith("vision:") for e in retriever.retrieve(f"What is the condition of {tag}?")))

        recall = round(true_positive / len(self._tags_with_evidence), 4) if self._tags_with_evidence else 0.0
        metrics = [
            NamedMetric("visual_evidence_recall", recall),
            NamedMetric("false_positive_count", float(false_positive)),
        ]
        total_samples = len(self._tags_with_evidence) + len(self._tags_without_evidence)
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_graph", total_samples)


class VisionLatencyBenchmark:
    """Purpose: measure per-operation wall-clock latency across the vision
    pipeline's classical-CV stages (OCR excluded — its ~15s one-time model
    load dominates and is already reported separately at startup)."""

    name = "vision_latency"

    def __init__(self, diagram_parser: ClassicalCVDiagramParser, object_detector: ClassicalCVObjectDetector, defect_detector: ClassicalCVDefectDetector, gauge_reader: GaugeReader):
        self._diagram_parser = diagram_parser
        self._object_detector = object_detector
        self._defect_detector = defect_detector
        self._gauge_reader = gauge_reader

    async def run(self) -> BenchmarkReport:
        import cv2

        start = time.perf_counter()
        diagram_path, photo_path, gauge_path = str(_bench_dir() / "_latency_diagram.png"), str(_bench_dir() / "_latency_photo.png"), str(_bench_dir() / "_latency_gauge.png")
        cv2.imwrite(diagram_path, _pid_array())
        cv2.imwrite(photo_path, _rust_patch_array())
        gauge_img, _, _ = _gauge_array(0.0)
        cv2.imwrite(gauge_path, gauge_img)

        t0 = time.perf_counter()
        self._diagram_parser.parse(diagram_path, "bench")
        diagram_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._object_detector.detect(photo_path, "bench")
        detection_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._defect_detector.detect(photo_path, "bench")
        defect_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._gauge_reader.read_all(gauge_path, "bench")
        gauge_latency = time.perf_counter() - t0

        metrics = [
            NamedMetric("diagram_parsing_seconds", round(diagram_latency, 4)),
            NamedMetric("object_detection_seconds", round(detection_latency, 4)),
            NamedMetric("defect_detection_seconds", round(defect_latency, 4)),
            NamedMetric("gauge_reading_seconds", round(gauge_latency, 4)),
        ]
        return BenchmarkReport(self.name, metrics, time.perf_counter() - start, "synthetic_images", 3)
