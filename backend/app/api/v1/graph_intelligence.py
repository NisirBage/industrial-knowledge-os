"""Graph Intelligence API (M9): dataset export, feature inspection,
embedding training, similarity search, explanations, link prediction, node
classification (registered alongside M6's classical models), and risk
propagation. Entirely independent of the chat orchestrator's request path —
same convention every milestone since M4 has followed.
"""

from fastapi import APIRouter, HTTPException, Request

from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.explanation import GraphExplanationService
from app.graph_intelligence.link_prediction import LinkPredictor
from app.graph_intelligence.orchestration import available_embedding_models
from app.graph_intelligence.propagation import PropagationType
from app.graph_intelligence.similarity import GraphSimilarityService
from app.prediction.models import PredictionTaskType
from app.schemas.ml import FeatureImportanceOut
from app.schemas.graph_intelligence import (
    EmbeddingTrainRequest,
    EmbeddingTrainResponse,
    ExplanationOut,
    GraphEdgeOut,
    GraphExportOut,
    GraphNodeOut,
    GraphSummaryOut,
    LinkPredictionCandidateOut,
    LinkPredictionRequest,
    NodeClassificationTrainRequest,
    NodeClassificationTrainResponse,
    NodeFeaturesOut,
    PropagationRequest,
    PropagationResultOut,
    PropagationStepOut,
    SimilarNodeOut,
)

router = APIRouter(prefix="/graph-intelligence", tags=["graph-intelligence"])


def _resolve_node_id(graph_repository, equipment_name: str) -> str:
    entity = graph_repository.find_entity(equipment_name)
    if entity is None:
        raise HTTPException(status_code=404, detail=f"No graph entity named {equipment_name!r}")
    return entity["id"]


@router.get("/summary", response_model=GraphSummaryOut)
async def get_summary(request: Request) -> GraphSummaryOut:
    graph_repository = request.app.state.graph_repository
    return GraphSummaryOut(
        node_count=graph_repository.node_count(),
        edge_count=graph_repository.edge_count(),
        node_types=sorted(graph_repository.distinct_entity_types()),
        relation_types=sorted(graph_repository.distinct_relation_types()),
    )


@router.get("/graph", response_model=GraphExportOut)
async def get_graph(request: Request) -> GraphExportOut:
    """Full node-link export for the Graph Visualization workspace (M9
    Component 8) — includes each node's community assignment so the
    frontend can color by community without a second round trip."""
    graph_repository = request.app.state.graph_repository
    feature_store = request.app.state.graph_feature_store
    builder = GraphDatasetBuilder(graph_repository)
    dataset = builder.build()
    structural = feature_store.structural_features(dataset)

    return GraphExportOut(
        nodes=[
            GraphNodeOut(
                id=n.id, type=n.type, canonical_name=n.canonical_name, confidence=n.confidence,
                community_id=structural.get(n.id, {}).get("community_id"),
            )
            for n in dataset.nodes
        ],
        edges=[GraphEdgeOut(source=e.source, target=e.target, relation_type=e.relation_type, confidence=e.confidence) for e in dataset.edges],
    )


@router.get("/features/{equipment_name}", response_model=NodeFeaturesOut)
async def get_node_features(equipment_name: str, request: Request) -> NodeFeaturesOut:
    graph_repository = request.app.state.graph_repository
    feature_store = request.app.state.graph_feature_store
    node_id = _resolve_node_id(graph_repository, equipment_name)

    dataset = GraphDatasetBuilder(graph_repository).build()
    structural = feature_store.structural_features(dataset)
    node = next(n for n in dataset.nodes if n.id == node_id)
    features = await feature_store.compute_node_features(node_id, node.canonical_name, node.type, structural.get(node_id, {}))
    return NodeFeaturesOut(**features.__dict__)


@router.get("/embeddings/models", response_model=list[str])
async def list_embedding_models() -> list[str]:
    return available_embedding_models()


@router.post("/embeddings/train", response_model=EmbeddingTrainResponse)
async def train_embeddings(payload: EmbeddingTrainRequest, request: Request) -> EmbeddingTrainResponse:
    trainer = request.app.state.graph_embedding_trainer
    try:
        result = await trainer.train(payload.model_name, dimensions=payload.dimensions)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request.app.state.graph_embedding_cache = result.embedding_result
    return EmbeddingTrainResponse(
        model_id=result.model_id,
        model_name=result.embedding_result.model_name,
        dimensions=result.embedding_result.dimensions,
        node_count=len(result.embedding_result.embeddings),
        training_seconds=result.embedding_result.training_seconds,
    )


async def _current_embedding(request: Request):
    cached = getattr(request.app.state, "graph_embedding_cache", None)
    if cached is not None:
        return cached
    inference_engine = request.app.state.graph_inference_engine
    loaded = await inference_engine.load_latest_embedding()
    if loaded is not None:
        request.app.state.graph_embedding_cache = loaded
    return loaded


@router.get("/similarity/{equipment_name}", response_model=list[SimilarNodeOut])
async def get_similar_nodes(equipment_name: str, request: Request, top_k: int = 5) -> list[SimilarNodeOut]:
    graph_repository = request.app.state.graph_repository
    node_id = _resolve_node_id(graph_repository, equipment_name)
    embedding_result = await _current_embedding(request)
    if embedding_result is None:
        raise HTTPException(status_code=400, detail="No embedding has been trained yet — POST /graph-intelligence/embeddings/train first.")

    similarity_service = GraphSimilarityService()
    neighbors = similarity_service.nearest_neighbors(embedding_result, node_id, k=top_k)
    id_to_name = {n["id"]: n["canonical_name"] for n in graph_repository.find_neighbors(node_id, depth=2)}
    id_to_name[node_id] = equipment_name
    return [SimilarNodeOut(node_id=nid, canonical_name=id_to_name.get(nid, nid), score=score) for nid, score in neighbors]


@router.get("/explain/{equipment_name}", response_model=ExplanationOut)
async def explain_node(equipment_name: str, request: Request) -> ExplanationOut:
    graph_repository = request.app.state.graph_repository
    node_id = _resolve_node_id(graph_repository, equipment_name)
    embedding_result = await _current_embedding(request)
    if embedding_result is None:
        raise HTTPException(status_code=400, detail="No embedding has been trained yet — POST /graph-intelligence/embeddings/train first.")

    feature_store = request.app.state.graph_feature_store
    dataset = GraphDatasetBuilder(graph_repository).build()
    structural = feature_store.structural_features(dataset)

    explanation_service = GraphExplanationService(graph_repository)
    explanation = explanation_service.explain_node(node_id, embedding_result, confidence=0.8, structural_features=structural)
    similarity_service = GraphSimilarityService()
    id_to_name = {n["id"]: n["canonical_name"] for n in graph_repository.find_neighbors(node_id, depth=1)}

    return ExplanationOut(
        node_id=explanation.node_id,
        nearest_neighbors=[
            SimilarNodeOut(node_id=nid, canonical_name=id_to_name.get(nid, nid), score=score) for nid, score in explanation.nearest_neighbors
        ],
        important_edges=explanation.important_edges,
        community_id=explanation.community_id,
        supporting_evidence=explanation.supporting_evidence,
        confidence=explanation.confidence,
        text=GraphExplanationService.to_text(explanation),
    )


@router.post("/link-prediction/candidates", response_model=list[LinkPredictionCandidateOut])
async def link_prediction_candidates(payload: LinkPredictionRequest, request: Request) -> list[LinkPredictionCandidateOut]:
    graph_repository = request.app.state.graph_repository
    node_id = _resolve_node_id(graph_repository, payload.equipment_name)
    embedding_result = await _current_embedding(request)
    if embedding_result is None:
        raise HTTPException(status_code=400, detail="No embedding has been trained yet — POST /graph-intelligence/embeddings/train first.")

    inference_engine = request.app.state.graph_inference_engine
    model = await inference_engine.train_link_predictor(embedding_result)

    dataset = GraphDatasetBuilder(graph_repository).build()
    predictor = LinkPredictor(graph_repository)
    candidates = predictor.top_candidates(node_id, dataset, embedding_result, model, top_k=payload.top_k)
    id_to_name = {n.id: n.canonical_name for n in dataset.nodes}

    return [
        LinkPredictionCandidateOut(
            source_id=c.source_id,
            target_id=c.target_id,
            target_canonical_name=id_to_name.get(c.target_id, c.target_id),
            confidence=c.confidence,
            embedding_similarity=c.embedding_similarity,
            explanation=c.explanation,
        )
        for c in candidates
    ]


@router.post("/node-classification/train", response_model=NodeClassificationTrainResponse)
async def train_node_classifier(payload: NodeClassificationTrainRequest, request: Request) -> NodeClassificationTrainResponse:
    try:
        task_type = PredictionTaskType(payload.task_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown task_type {payload.task_type!r}. Valid: {[t.value for t in PredictionTaskType]}") from exc

    embedding_result = await _current_embedding(request)
    if embedding_result is None:
        raise HTTPException(status_code=400, detail="No embedding has been trained yet — POST /graph-intelligence/embeddings/train first.")

    inference_engine = request.app.state.graph_inference_engine
    feature_extractor = request.app.state.feature_extractor
    rule_based_models = request.app.state.rule_based_models
    rule_model = rule_based_models.get(task_type)
    if rule_model is None:
        raise HTTPException(status_code=400, detail=f"No rule-based baseline registered for task_type {task_type.value!r}.")

    graph_model = await inference_engine.train_node_classifier(
        task_type, embedding_result, feature_extractor, rule_model, payload.equipment_names
    )
    if graph_model is None:
        return NodeClassificationTrainResponse(registered=False, task_type=task_type.value, sample_size=0)

    prediction_engine = request.app.state.prediction_engine
    prediction_engine.register_model(task_type, graph_model)

    # Same Explainer M6's classical models use (app/ml/explainability) — a
    # graph model's estimator is a real RandomForestRegressor, so
    # feature_importances_ works unmodified. Feature names are embedding
    # dimensions (dim_0, dim_1, ...) rather than domain features, an honest
    # reflection of what a graph-embedding model actually learns from.
    explainer = request.app.state.ml_explainer
    trained_model = graph_model.trained_model
    importance = explainer.feature_importance(trained_model.estimator, trained_model.feature_names)

    return NodeClassificationTrainResponse(
        registered=True,
        task_type=task_type.value,
        algorithm=trained_model.algorithm,
        sample_size=len(payload.equipment_names),
        feature_importance=[FeatureImportanceOut(feature_name=f.feature_name, importance=f.importance) for f in importance],
    )


@router.post("/propagation/simulate", response_model=PropagationResultOut)
async def simulate_propagation(payload: PropagationRequest, request: Request) -> PropagationResultOut:
    graph_repository = request.app.state.graph_repository
    node_id = _resolve_node_id(graph_repository, payload.equipment_name)

    try:
        propagation_type = PropagationType(payload.propagation_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"Unknown propagation_type {payload.propagation_type!r}. Valid: {[t.value for t in PropagationType]}"
        ) from exc

    propagation_engine = request.app.state.graph_propagation_engine
    result = propagation_engine.propagate(
        node_id, propagation_type, initial_risk=payload.initial_risk, max_depth=payload.max_depth, decay_factor=payload.decay_factor
    )

    return PropagationResultOut(
        source_node_id=result.source_node_id,
        propagation_type=result.propagation_type.value,
        steps=[
            PropagationStepOut(node_id=s.node_id, canonical_name=s.canonical_name, depth=s.depth, propagated_risk=s.propagated_risk, path=s.path)
            for s in result.steps
        ],
    )
