"""Composition root: wires every layer (config -> repositories -> extraction ->
retrieval -> planning -> evidence/reasoning/recommendation -> agents ->
orchestrator -> services) into the FastAPI app once at startup. Each
dependency is constructed here and only here, so any layer can be swapped
without touching the modules that consume it.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.reasoning_agent import ReasoningAgent
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
from app.feedback.engine import FeedbackEngine
from app.features.extractor import RuleBasedFeatureExtractor
from app.graph.knowledge_graph_builder import KnowledgeGraphBuilder
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
from app.retrieval.bm25_index import Bm25Index
from app.retrieval.graph_expansion import GraphExpansionStage
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.services.chat_service import ChatService
from app.services.ingestion_service import IngestionService
from app.services.llm_provider import build_llm_provider
from app.timeline.engine import TimelineEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()

    embedder = build_embedder(settings.embedding_model)
    vector_repository = build_vector_repository(settings.qdrant_url, settings.qdrant_collection, settings.qdrant_local_path)
    graph_repository = build_graph_repository(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    graph_repository.ensure_constraints()
    metadata_repository = MetadataRepository(settings.metadata_db_url)
    await metadata_repository.init_models()
    bm25_index = Bm25Index()

    llm_provider = build_llm_provider(settings.llm_provider, settings.anthropic_api_key, settings.llm_model)

    primary_extractor, supplement_extractor = build_entity_extractors(settings.gliner_model, settings.gliner_threshold)
    entity_validator = LlmEntityValidator(llm_provider, enabled=settings.enable_llm_entity_validation)
    entity_extraction_service = EntityExtractionService(primary_extractor, supplement_extractor, entity_validator)

    relation_extractor = build_relation_extractor()
    knowledge_graph_builder = KnowledgeGraphBuilder(graph_repository)

    reranker = build_reranker(settings.reranker_model)
    graph_expansion = GraphExpansionStage(entity_extraction_service, graph_repository)
    retriever = HybridRetriever(embedder, vector_repository, bm25_index, graph_expansion, reranker)

    task_graph_planner = TaskGraphPlanner()
    task_graph_executor = TaskGraphExecutor(retriever, graph_repository)

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
    prediction_engine = PredictionEngine(feature_extractor, build_rule_based_models(), learning_repository)
    feedback_engine = FeedbackEngine(learning_repository)

    app.state.learning_repository = learning_repository
    app.state.timeline_engine = timeline_engine
    app.state.feature_extractor = feature_extractor
    app.state.prediction_engine = prediction_engine
    app.state.feedback_engine = feedback_engine

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

    yield

    graph_repository.close()


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
