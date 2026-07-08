"""Research API (M5): experiment history, on-demand benchmark runs, A/B
comparisons, and confidence calibration — the data surface the Research
Dashboard (frontend) reads from. Entirely independent of the chat
orchestrator; nothing here is on the M3 request path.
"""

from fastapi import APIRouter, HTTPException, Request

from app.documents.chunking import Chunk
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.evaluators import EntityExtractionEvaluator
from app.ontology.labels import EntityLabel, RelationType
from app.research.ab_testing.comparator import ABComparator
from app.research.benchmarks.models import BenchmarkReport
from app.research.benchmarks.runners import (
    EntityExtractionBenchmark,
    KnowledgeGraphBenchmark,
    RelationExtractionBenchmark,
)
from app.research.benchmarks.vision_runners import (
    DefectDetectionBenchmark,
    DiagramParsingBenchmark,
    EquipmentRecognitionBenchmark,
    GaugeReadingBenchmark,
    MultimodalRetrievalBenchmark,
    OCRBenchmark,
    VisionLatencyBenchmark,
)
from app.research.benchmarks.graph_runners import (
    GraphEmbeddingBenchmark,
    LinkPredictionBenchmark,
    NodeClassificationBenchmark,
)
from app.research.datasets.generator import SyntheticDatasetGenerator
from app.research.experiments.manager import ExperimentManager
from app.research.experiments.models import Experiment
from app.research.model_registry import entity_extractor_version
from app.schemas.research import (
    ABTestRequest,
    ABTestResultOut,
    BenchmarkReportOut,
    BenchmarkRunRequest,
    CalibrationBucketOut,
    CalibrationOut,
    ComponentVersionOut,
    ExperimentOut,
    NamedMetricOut,
)

router = APIRouter(prefix="/research", tags=["research"])

_AUTO_RUN_BENCHMARKS = {
    "entity_extraction",
    "relation_extraction",
    "knowledge_graph",
    "vision_ocr",
    "vision_diagram_parsing",
    "vision_equipment_recognition",
    "vision_gauge_reading",
    "vision_defect_detection",
    "vision_multimodal_retrieval",
    "vision_latency",
    "graph_embedding_node2vec",
    "graph_embedding_deepwalk",
    "graph_embedding_gcn",
    "graph_embedding_graphsage",
    "graph_embedding_gat",
    "graph_link_prediction",
    "graph_node_classification",
}

# M7 vision benchmarks generate their own small deterministic synthetic
# images/graphs (see vision_runners.py) rather than consuming the text
# SyntheticDatasetGenerator's output — so this dispatch skips that step for
# them, unlike the three text benchmarks above them.
_VISION_BENCHMARK_NAMES = {
    "vision_ocr",
    "vision_diagram_parsing",
    "vision_equipment_recognition",
    "vision_gauge_reading",
    "vision_defect_detection",
    "vision_multimodal_retrieval",
    "vision_latency",
}

# M9 graph benchmarks likewise generate their own small deterministic
# synthetic graph (see graph_runners.py's _synthetic_graph_repo) rather than
# the live production graph or the text SyntheticDatasetGenerator.
_GRAPH_BENCHMARK_NAMES = {
    "graph_embedding_node2vec",
    "graph_embedding_deepwalk",
    "graph_embedding_gcn",
    "graph_embedding_graphsage",
    "graph_embedding_gat",
    "graph_link_prediction",
    "graph_node_classification",
}


def _experiment_to_out(experiment: Experiment) -> ExperimentOut:
    def _versions(versions):
        return [ComponentVersionOut(kind=v.kind.value, name=v.name, version=v.version, config=v.config) for v in versions]

    return ExperimentOut(
        experiment_id=experiment.experiment_id,
        name=experiment.name,
        description=experiment.description,
        configuration=experiment.configuration,
        model_versions=_versions(experiment.model_versions),
        retriever_versions=_versions(experiment.retriever_versions),
        graph_versions=_versions(experiment.graph_versions),
        reranker=experiment.reranker,
        ontology_version=experiment.ontology_version,
        dataset=experiment.dataset,
        timestamp=experiment.timestamp,
        status=experiment.status.value,
        metrics=experiment.metrics,
        duration=experiment.duration,
        random_seed=experiment.random_seed,
        git_commit=experiment.git_commit,
        error=experiment.error,
    )


def _report_to_out(report: BenchmarkReport) -> BenchmarkReportOut:
    return BenchmarkReportOut(
        benchmark_name=report.benchmark_name,
        metrics=[NamedMetricOut(name=m.name, value=m.value) for m in report.metrics],
        duration_seconds=report.duration_seconds,
        dataset_name=report.dataset_name,
        sample_size=report.sample_size,
        timestamp=report.timestamp,
    )


@router.get("/experiments", response_model=list[ExperimentOut])
async def list_experiments(request: Request) -> list[ExperimentOut]:
    manager: ExperimentManager = request.app.state.experiment_manager
    return [_experiment_to_out(e) for e in await manager.list()]


@router.get("/experiments/{experiment_id}", response_model=ExperimentOut)
async def get_experiment(experiment_id: str, request: Request) -> ExperimentOut:
    manager: ExperimentManager = request.app.state.experiment_manager
    experiment = await manager.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"No experiment with id {experiment_id!r}")
    return _experiment_to_out(experiment)


@router.post("/benchmarks/{benchmark_name}/run", response_model=BenchmarkReportOut)
async def run_benchmark(benchmark_name: str, payload: BenchmarkRunRequest, request: Request) -> BenchmarkReportOut:
    if benchmark_name not in _AUTO_RUN_BENCHMARKS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"'{benchmark_name}' isn't runnable from a fresh synthetic dataset alone "
                f"(it needs already-ingested documents or live equipment data). "
                f"Auto-runnable: {sorted(_AUTO_RUN_BENCHMARKS)}. Other benchmarks are available "
                f"programmatically via app.research.benchmarks.runners."
            ),
        )

    entity_extraction_service = request.app.state.entity_extraction_service
    relation_extractor = request.app.state.relation_extractor
    graph_repository = request.app.state.graph_repository

    if benchmark_name in _GRAPH_BENCHMARK_NAMES:
        if benchmark_name.startswith("graph_embedding_"):
            model_name = benchmark_name.removeprefix("graph_embedding_")
            report = await GraphEmbeddingBenchmark(model_name).run()
        elif benchmark_name == "graph_link_prediction":
            report = await LinkPredictionBenchmark().run()
        else:
            report = await NodeClassificationBenchmark().run()
    elif benchmark_name in _VISION_BENCHMARK_NAMES:
        if benchmark_name == "vision_ocr":
            report = await OCRBenchmark(request.app.state.vision_ocr_provider).run()
        elif benchmark_name == "vision_diagram_parsing":
            report = await DiagramParsingBenchmark(request.app.state.vision_diagram_parser).run()
        elif benchmark_name == "vision_equipment_recognition":
            report = await EquipmentRecognitionBenchmark(request.app.state.vision_object_detector).run()
        elif benchmark_name == "vision_gauge_reading":
            report = await GaugeReadingBenchmark(request.app.state.vision_gauge_reader).run()
        elif benchmark_name == "vision_defect_detection":
            report = await DefectDetectionBenchmark(request.app.state.vision_defect_detector).run()
        elif benchmark_name == "vision_multimodal_retrieval":
            report = await MultimodalRetrievalBenchmark().run()
        else:
            report = await VisionLatencyBenchmark(
                request.app.state.vision_diagram_parser,
                request.app.state.vision_object_detector,
                request.app.state.vision_defect_detector,
                request.app.state.vision_gauge_reader,
            ).run()
    else:
        generator = SyntheticDatasetGenerator(seed=payload.seed)
        documents = generator.generate_dataset(payload.sample_size)

        if benchmark_name == "entity_extraction":
            report = await EntityExtractionBenchmark(entity_extraction_service, documents).run()
        elif benchmark_name == "relation_extraction":
            report = await RelationExtractionBenchmark(entity_extraction_service, relation_extractor, documents).run()
        else:
            report = await KnowledgeGraphBenchmark(graph_repository, len(EntityLabel), len(RelationType)).run()

    experiment_manager: ExperimentManager = request.app.state.experiment_manager
    experiment = await experiment_manager.start(
        name=f"benchmark:{benchmark_name}",
        description=f"Auto-run benchmark for {benchmark_name} on a synthetic dataset",
        configuration={"sample_size": str(payload.sample_size), "seed": str(payload.seed)},
        model_versions=[entity_extractor_version("entity_extraction_service", "v1")],
        retriever_versions=[],
        graph_versions=[],
        reranker=request.app.state.settings.reranker_model,
        ontology_version="1.0",
        dataset="synthetic",
        random_seed=payload.seed,
    )
    await experiment_manager.complete(experiment.experiment_id, {m.name: m.value for m in report.metrics})

    return _report_to_out(report)


@router.get("/benchmarks/{benchmark_name}/history", response_model=list[ExperimentOut])
async def benchmark_history(benchmark_name: str, request: Request) -> list[ExperimentOut]:
    manager: ExperimentManager = request.app.state.experiment_manager
    experiments = await manager.list()
    filtered = [e for e in experiments if e.name == f"benchmark:{benchmark_name}"]
    return [_experiment_to_out(e) for e in filtered]


@router.post("/ab-test/entity-extraction", response_model=ABTestResultOut)
async def run_entity_extraction_ab_test(payload: ABTestRequest, request: Request) -> ABTestResultOut:
    """Regex-only vs the production entity extraction service (GLiNER+regex,
    or regex-only if GLiNER couldn't load) — the exact comparison the M5
    spec's A/B examples call out ("Regex vs GLiNER")."""
    production_service = request.app.state.entity_extraction_service
    regex_only_service = EntityExtractionService(RegexEntityExtractor())

    generator = SyntheticDatasetGenerator(seed=payload.seed)
    documents = generator.generate_dataset(payload.sample_size)
    evaluator = EntityExtractionEvaluator()

    arm_a_scores: list[float] = []
    arm_b_scores: list[float] = []
    for doc in documents:
        chunk = Chunk(id=doc.doc_id, document_id=doc.doc_id, text=doc.text, page_number=1, chunk_index=0)
        predicted_a = [(e.label.value, e.canonical_name) for e in regex_only_service.extract(chunk)]
        predicted_b = [(e.label.value, e.canonical_name) for e in production_service.extract(chunk)]

        f1_a = next(m.value for m in evaluator.evaluate(predicted_a, doc.ground_truth_entities) if m.name == "f1")
        f1_b = next(m.value for m in evaluator.evaluate(predicted_b, doc.ground_truth_entities) if m.name == "f1")
        arm_a_scores.append(f1_a)
        arm_b_scores.append(f1_b)

    result = ABComparator().compare("regex_only", "production (gliner+regex)", "f1", arm_a_scores, arm_b_scores)
    return ABTestResultOut(
        arm_a_name=result.arm_a_name,
        arm_b_name=result.arm_b_name,
        metric_name=result.metric_name,
        mean_a=result.mean_a,
        mean_b=result.mean_b,
        mean_delta=result.mean_delta,
        t_statistic=result.t_statistic,
        p_value=result.p_value,
        winner=result.winner,
    )


@router.get("/calibration", response_model=CalibrationOut)
async def get_calibration(request: Request) -> CalibrationOut:
    learning_repository = request.app.state.learning_repository
    buckets = await learning_repository.confidence_calibration_buckets()
    return CalibrationOut(buckets=[CalibrationBucketOut(**b) for b in buckets])
