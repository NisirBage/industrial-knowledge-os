from app.research.benchmarks.vision_runners import (
    DefectDetectionBenchmark,
    DiagramParsingBenchmark,
    EquipmentRecognitionBenchmark,
    GaugeReadingBenchmark,
    MultimodalRetrievalBenchmark,
    OCRBenchmark,
    VisionLatencyBenchmark,
)
from app.vision.defect.detector import ClassicalCVDefectDetector
from app.vision.detection.detector import ClassicalCVObjectDetector
from app.vision.diagram.parser import ClassicalCVDiagramParser
from app.vision.gauge.reader import GaugeReader
from app.vision.ocr.provider import build_ocr_provider


async def test_diagram_parsing_benchmark_finds_all_three_symbol_types():
    report = await DiagramParsingBenchmark(ClassicalCVDiagramParser()).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["symbol_type_recall"] == 1.0
    assert metrics["total_symbols_detected"] > 0
    assert report.benchmark_name == "vision_diagram_parsing"


async def test_equipment_recognition_benchmark_finds_the_foreground_region_every_time():
    report = await EquipmentRecognitionBenchmark(ClassicalCVObjectDetector(), num_samples=3).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["region_detection_recall"] == 1.0
    assert report.sample_size == 3


async def test_gauge_reading_benchmark_reports_low_error_for_known_angles():
    report = await GaugeReadingBenchmark(GaugeReader()).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["mean_absolute_error"] < 0.05
    assert metrics["valid_reading_rate"] == 1.0


async def test_defect_detection_benchmark_finds_both_corrosion_and_crack():
    report = await DefectDetectionBenchmark(ClassicalCVDefectDetector()).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["corrosion_detected"] == 1.0
    assert metrics["crack_detected"] == 1.0
    assert metrics["defect_detection_recall"] == 1.0


async def test_vision_latency_benchmark_reports_a_positive_duration_for_every_stage():
    report = await VisionLatencyBenchmark(
        ClassicalCVDiagramParser(), ClassicalCVObjectDetector(), ClassicalCVDefectDetector(), GaugeReader()
    ).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert all(value >= 0.0 for value in metrics.values())
    assert set(metrics) == {
        "diagram_parsing_seconds",
        "object_detection_seconds",
        "defect_detection_seconds",
        "gauge_reading_seconds",
    }


async def test_multimodal_retrieval_benchmark_reports_perfect_recall_on_its_synthetic_graph():
    report = await MultimodalRetrievalBenchmark().run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["visual_evidence_recall"] == 1.0
    assert metrics["false_positive_count"] == 0.0


async def test_multimodal_retrieval_benchmark_sample_size_matches_tag_counts():
    report = await MultimodalRetrievalBenchmark(tags_with_evidence=["P-1"], tags_without_evidence=["T-1", "T-2"]).run()

    assert report.sample_size == 3


async def test_ocr_benchmark_reads_rendered_tags_or_degrades_to_zero_not_a_crash():
    provider = build_ocr_provider()
    report = await OCRBenchmark(provider, tags=["P-101"]).run()

    metrics = {m.name: m.value for m in report.metrics}
    assert metrics["tag_read_accuracy"] in (0.0, 1.0)
    assert 0.0 <= metrics["mean_confidence"] <= 1.0
