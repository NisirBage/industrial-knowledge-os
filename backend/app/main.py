"""Composition root: wires every layer (config -> repositories -> extraction ->
retrieval -> planning -> evidence/reasoning/recommendation -> agents ->
orchestrator -> services) into the FastAPI app once at startup. Each
dependency is constructed here and only here, so any layer can be swapped
without touching the modules that consume it.
"""

# Windows-specific DLL load-order bug (M9): importing `neo4j` before `torch`
# in the same process reliably breaks torch's native DLL init
# (`OSError: [WinError 1114] ... c10.dll`) on this platform — reproduced
# with a two-line repro (`from neo4j import GraphDatabase; import torch`
# fails; swapping the order works every time). Neither library is at fault
# in isolation; some shared native dependency (likely OpenSSL) conflicts
# depending on load order. Importing torch first, here, before
# `build_graph_repository` (which imports neo4j) ever runs, is the fix.
import torch  # noqa: F401

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.api.v1.router import api_router
from app.contradiction.detector import ContradictionDetector
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.embeddings.embedder import build_embedder
from app.entity_extraction.factory import build_entity_extractors
from app.entity_extraction.llm_validator import LlmEntityValidator
from app.entity_extraction.service import EntityExtractionService
from app.evidence.fusion import EvidenceFusionEngine
from app.evidence.graph import EvidenceGraphBuilder
from app.features.extractor import RuleBasedFeatureExtractor
from app.feedback.engine import FeedbackEngine
from app.graph.knowledge_graph_builder import KnowledgeGraphBuilder
from app.graph_intelligence.feature_store.store import GraphFeatureStore
from app.graph_intelligence.orchestration import GraphEmbeddingTrainer, GraphInferenceEngine
from app.graph_intelligence.propagation import GraphPropagationEngine
from app.ml.explainability.explainer import Explainer
from app.ml.feature_store.store import FeatureStore
from app.ml.monitoring.monitor import ModelMonitor
from app.ml.registry.repository import SqlModelRegistry
from app.ml.training.trainer import SklearnTrainer
from app.narration.narrator import LLMNarrator
from app.orchestrator.graph import build_orchestrator
from app.planning.executor import TaskGraphExecutor
from app.planning.planner import TaskGraphPlanner
from app.prediction.engine import PredictionEngine
from app.prediction.rule_models import build_rule_based_models
from app.reasoning.reasoning_graph_builder import ReasoningGraphBuilder
from app.recommendation.engine import RecommendationEngine
from app.relation_extraction.factory import build_relation_extractor
from app.repositories.graph_repository import build_graph_repository
from app.repositories.learning_repository import SqlLearningRepository
from app.repositories.metadata_repository import MetadataRepository
from app.repositories.vector_repository import build_vector_repository
from app.research.experiments.manager import ExperimentManager
from app.research.experiments.repository import SqlExperimentTracker
from app.caching.lru_cache import LRUTTLCache
from app.retrieval.bm25_index import Bm25Index
from app.retrieval.graph_expansion import GraphExpansionStage
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.llm_provider import build_llm_provider
from app.timeline.engine import TimelineEngine
from app.vision.defect.detector import ClassicalCVDefectDetector
from app.vision.detection.detector import ClassicalCVObjectDetector
from app.vision.diagram.parser import ClassicalCVDiagramParser
from app.vision.gauge.reader import GaugeReader
from app.vision.graph.builder import VisionKnowledgeGraphBuilder
from app.vision.ingestion.repository import VisionMetadataRepository
from app.vision.ingestion.service import VisionIngestionService
from app.vision.ocr.provider import build_ocr_provider
from app.vision.retrieval.multimodal import MultimodalRetriever

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    startup_start = time.perf_counter()

    # These six checks are independent at construction time (each either
    # connects to/loads a real dependency or falls back) — running them
    # concurrently via threads bounds cold-start time by the slowest single
    # check instead of their sum, in principle. v1.0.1 hardening: measured
    # real-world improvement in this environment was ~20s concurrent vs.
    # ~25-31s sequential (both across repeated runs) — a real but modest
    # gain, well short of the theoretical best case, because the checks are
    # network-bound (HTTP/DNS resolution attempts) and their individual
    # durations vary run to run rather than summing predictably. See
    # docs/architecture/deployment.md#startup-performance for the honest
    # measured numbers, including one anomalous ~39s run. M7 adds a sixth:
    # the OCR provider (EasyOCR model load), same degrade-not-crash pattern.
    embedder_task = asyncio.to_thread(build_embedder, settings.embedding_model)
    vector_repository_task = asyncio.to_thread(
        build_vector_repository, settings.qdrant_url, settings.qdrant_collection, settings.qdrant_local_path
    )
    graph_repository_task = asyncio.to_thread(
        build_graph_repository, settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
    )
    entity_extractors_task = asyncio.to_thread(build_entity_extractors, settings.gliner_model, settings.gliner_threshold)
    reranker_task = asyncio.to_thread(build_reranker, settings.reranker_model)
    ocr_provider_task = asyncio.to_thread(build_ocr_provider)

    (
        embedder,
        vector_repository,
        graph_repository,
        (primary_extractor, supplement_extractor),
        reranker,
        ocr_provider,
    ) = await asyncio.gather(
        embedder_task, vector_repository_task, graph_repository_task, entity_extractors_task, reranker_task, ocr_provider_task
    )
    graph_repository.ensure_constraints()

    logger.info("Independent dependency checks (embedder/qdrant/neo4j/gliner/reranker) completed in %.2fs", time.perf_counter() - startup_start)

    metadata_repository = MetadataRepository(settings.metadata_db_url)
    await metadata_repository.init_models()
    bm25_index = Bm25Index()

    llm_provider = build_llm_provider(settings.llm_provider, settings.anthropic_api_key, settings.llm_model)

    entity_validator = LlmEntityValidator(llm_provider, enabled=settings.enable_llm_entity_validation)
    entity_extraction_service = EntityExtractionService(primary_extractor, supplement_extractor, entity_validator)

    relation_extractor = build_relation_extractor()
    knowledge_graph_builder = KnowledgeGraphBuilder(graph_repository)

    graph_expansion = GraphExpansionStage(entity_extraction_service, graph_repository)

    # O1 Component 6: retrieval caches — explicit, per-instance objects (no
    # module-level singletons), injected into HybridRetriever like every
    # other collaborator. TTLs are short (query embeddings/expansions can go
    # stale as new documents are ingested) and sizes are modest — this is a
    # per-process cache, not a shared/distributed one.
    retrieval_embedding_cache = LRUTTLCache(max_size=512, default_ttl=600)
    retrieval_expansion_cache = LRUTTLCache(max_size=512, default_ttl=300)
    retrieval_reranker_cache = LRUTTLCache(max_size=2048, default_ttl=300)
    retrieval_result_cache = LRUTTLCache(max_size=256, default_ttl=120)

    # O1 Component 4: two *separate* thread pools, not one shared pool.
    # TaskGraphExecutor's node-level parallelism submits work that itself
    # calls HybridRetriever.retrieve(), which (given an executor) submits
    # further sub-tasks and blocks on their result — nesting that submission
    # pattern onto a single shared pool risks deadlock if every worker ends
    # up blocked waiting on sub-tasks queued behind them with no free worker
    # left to run those sub-tasks. Two distinct pools have no such risk: a
    # worker blocked in pool A is never one of pool B's own worker threads.
    retrieval_fetch_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval-fetch")
    task_graph_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="task-graph")

    retriever = HybridRetriever(
        embedder,
        vector_repository,
        bm25_index,
        graph_expansion,
        reranker,
        embedding_cache=retrieval_embedding_cache,
        expansion_cache=retrieval_expansion_cache,
        reranker_cache=retrieval_reranker_cache,
        result_cache=retrieval_result_cache,
        executor=retrieval_fetch_pool,
    )

    task_graph_planner = TaskGraphPlanner()
    task_graph_executor = TaskGraphExecutor(retriever, graph_repository, executor=task_graph_pool)

    evidence_fusion_engine = EvidenceFusionEngine()
    contradiction_detector = ContradictionDetector()
    evidence_graph_builder = EvidenceGraphBuilder()
    reasoning_graph_builder = ReasoningGraphBuilder()
    recommendation_engine = RecommendationEngine(llm_provider)
    narrator = LLMNarrator(llm_provider)

    orchestrator = build_orchestrator(
        planner_agent=PlannerAgent(entity_extraction_service, task_graph_planner),
        retrieval_agent=RetrievalAgent(task_graph_executor),
        knowledge_graph_agent=KnowledgeGraphAgent(graph_expansion),
        evidence_fusion_agent=EvidenceFusionAgent(evidence_fusion_engine),
        reasoning_agent=ReasoningAgent(contradiction_detector, evidence_graph_builder, reasoning_graph_builder),
        recommendation_agent=RecommendationAgent(recommendation_engine),
        verifier_agent=VerifierAgent(),
        report_agent=ReportAgent(narrator),
    )

    app.state.embedder = embedder
    app.state.ingestion_service = IngestionService(
        embedder=embedder,
        vector_repository=vector_repository,
        metadata_repository=metadata_repository,
        bm25_index=bm25_index,
        entity_extraction_service=entity_extraction_service,
        relation_extractor=relation_extractor,
        knowledge_graph_builder=knowledge_graph_builder,
    )
    app.state.chat_service = ChatService(orchestrator)
    app.state.metadata_repository = metadata_repository

    # Learning Layer (M4): independent of the chat orchestrator above by
    # design — see project memory / M4 scope decision. Its own repository,
    # own API routes, no changes to the M3 pipeline.
    learning_repository = SqlLearningRepository(settings.learning_db_url)
    await learning_repository.init_models()
    timeline_engine = TimelineEngine(graph_repository)
    feature_extractor = RuleBasedFeatureExtractor(graph_repository, timeline_engine, learning_repository)
    # Kept as its own dict, not just what's passed into PredictionEngine —
    # M6/M9 both swap trained models into the engine at runtime via
    # register_model(), which would otherwise make "the rule baseline"
    # unrecoverable once a task_type's slot is occupied by a trained model.
    # Graph Intelligence (M9) needs the *original* rule-based model as its
    # training target regardless of what's currently live.
    rule_based_models = build_rule_based_models()
    prediction_engine = PredictionEngine(feature_extractor, dict(rule_based_models), learning_repository)
    feedback_engine = FeedbackEngine(learning_repository)

    app.state.learning_repository = learning_repository
    app.state.timeline_engine = timeline_engine
    app.state.feature_extractor = feature_extractor
    app.state.prediction_engine = prediction_engine
    app.state.feedback_engine = feedback_engine
    app.state.rule_based_models = rule_based_models

    # Research Layer (M5): also independent of the chat orchestrator. Reuses
    # the same entity_extraction_service/relation_extractor/graph_repository
    # instances above so benchmarks measure the exact components chat uses.
    experiment_tracker = SqlExperimentTracker(settings.experiment_db_url)
    await experiment_tracker.init_models()
    experiment_manager = ExperimentManager(experiment_tracker)

    app.state.settings = settings
    app.state.entity_extraction_service = entity_extraction_service
    app.state.relation_extractor = relation_extractor
    app.state.graph_repository = graph_repository
    app.state.retriever = retriever
    app.state.retrieval_embedding_cache = retrieval_embedding_cache
    app.state.retrieval_expansion_cache = retrieval_expansion_cache
    app.state.retrieval_reranker_cache = retrieval_reranker_cache
    app.state.retrieval_result_cache = retrieval_result_cache
    app.state.retrieval_fetch_pool = retrieval_fetch_pool
    app.state.task_graph_pool = task_graph_pool
    app.state.active_ranker = None
    app.state.merge_history = {}
    app.state.experiment_manager = experiment_manager

    # ML Laboratory (M6): also independent of the chat orchestrator. Trained
    # models are swapped into prediction_engine (M4) at runtime via
    # register_model — everything else about the chat/prediction API is
    # untouched until a model is actually trained.
    ml_model_registry = SqlModelRegistry(settings.ml_registry_db_url, Path(settings.ml_models_dir))
    await ml_model_registry.init_models()

    app.state.feature_store = FeatureStore(feature_extractor)
    app.state.ml_trainer = SklearnTrainer()
    app.state.ml_explainer = Explainer()
    app.state.model_monitor = ModelMonitor(learning_repository)
    app.state.model_registry = ml_model_registry

    # Vision (M7): also independent of the chat orchestrator. Reuses the same
    # graph_repository and entity_extraction_service instances above so
    # visual evidence lands in the exact graph text retrieval already reads,
    # and multimodal queries resolve entities with the exact same service —
    # "never bypass the Knowledge Graph" from the M7 spec, satisfied
    # structurally rather than by convention (same instances, not new ones).
    vision_metadata_repository = VisionMetadataRepository(settings.vision_db_url)
    await vision_metadata_repository.init_models()

    app.state.vision_metadata_repository = vision_metadata_repository
    app.state.vision_ingestion_service = VisionIngestionService(vision_metadata_repository, Path(settings.vision_images_dir))
    app.state.vision_ocr_provider = ocr_provider
    app.state.vision_diagram_parser = ClassicalCVDiagramParser()
    app.state.vision_object_detector = ClassicalCVObjectDetector()
    app.state.vision_gauge_reader = GaugeReader()
    app.state.vision_defect_detector = ClassicalCVDefectDetector()
    app.state.vision_graph_builder = VisionKnowledgeGraphBuilder(graph_repository)
    app.state.multimodal_retriever = MultimodalRetriever(retriever, graph_repository, entity_extraction_service)

    # Graph Intelligence (M9): also independent of the chat orchestrator.
    # Reuses graph_repository, learning_repository, and ml_model_registry —
    # every trained artifact (embeddings, link predictors, graph-embedding
    # PredictionModels) registers into the *same* SqlModelRegistry M6 built,
    # not a parallel one. graph_embedding_cache starts empty; the first
    # POST /graph-intelligence/embeddings/train call (or a restart that
    # finds a prior run in the registry) populates it.
    app.state.graph_feature_store = GraphFeatureStore(graph_repository, learning_repository)
    app.state.graph_embedding_trainer = GraphEmbeddingTrainer(graph_repository, ml_model_registry)
    app.state.graph_inference_engine = GraphInferenceEngine(graph_repository, ml_model_registry)
    app.state.graph_propagation_engine = GraphPropagationEngine(graph_repository)
    app.state.graph_embedding_cache = None

    logger.info("Application startup complete in %.2fs", time.perf_counter() - startup_start)

    yield

    graph_repository.close()
    retrieval_fetch_pool.shutdown(wait=False)
    task_graph_pool.shutdown(wait=False)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Industrial Knowledge Intelligence OS", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()
