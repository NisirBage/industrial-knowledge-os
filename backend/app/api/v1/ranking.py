"""Learning-to-Rank API (O1 Component 1): trains a real LightGBM ranker
against labeled queries built directly from the live Knowledge Graph — every
node's canonical name becomes a "Tell me about {name}" query, labeled
relevant against whatever documents `GraphRepository.documents_mentioning()`
already knows mention that entity. No synthetic-document ingestion pipeline
is built for this endpoint: the graph already knows which documents are
relevant to which entities (M3), so training data comes from that, not a
parallel labeling mechanism.
"""

from fastapi import APIRouter, HTTPException, Request

from app.ml.explainability.explainer import Explainer
from app.ml.registry.repository import SqlModelRegistry, hash_dataframe
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ranking.features import RankingFeatures
from app.retrieval.ranking.lightgbm_ranker import LightGBMRanker
from app.retrieval.ranking.trainer import LTRTrainer
from app.schemas.ml import FeatureImportanceOut
from app.schemas.ranking import RankingStatusOut, RankingTrainRequest, RankingTrainResponse

router = APIRouter(prefix="/ranking", tags=["ranking"])

_TASK_TYPE = "retrieval_ranking"


def _labeled_queries_from_graph(graph_repository, limit: int) -> list[tuple[str, set[str]]]:
    queries: list[tuple[str, set[str]]] = []
    for node in graph_repository.all_nodes()[:limit]:
        name = node.get("canonical_name")
        if not name:
            continue
        relevant_document_ids = {doc["document_id"] for doc in graph_repository.documents_mentioning(name)}
        if relevant_document_ids:
            queries.append((f"Tell me about {name}", relevant_document_ids))
    return queries


@router.post("/train", response_model=RankingTrainResponse)
async def train_ranker(payload: RankingTrainRequest, request: Request) -> RankingTrainResponse:
    graph_repository = request.app.state.graph_repository
    retriever: HybridRetriever = request.app.state.retriever

    labeled_queries = _labeled_queries_from_graph(graph_repository, payload.sample_size)
    if len(labeled_queries) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough graph entities with known relevant documents to train a ranker "
                f"(found {len(labeled_queries)}, need at least 2) — ingest more documents first."
            ),
        )

    trainer = LTRTrainer(retriever)
    ranker = trainer.fit(labeled_queries, n_estimators=payload.n_estimators)

    if not ranker.is_trained:
        return RankingTrainResponse(trained=False, algorithm="reciprocal_rank_fusion_fallback", sample_size=len(labeled_queries))

    model_registry: SqlModelRegistry = request.app.state.model_registry
    feature_names = RankingFeatures.feature_names()
    explainer = Explainer()
    importance = explainer.feature_importance(ranker.estimator, feature_names)

    import pandas as pd

    await model_registry.register(
        estimator=ranker.estimator,
        task_type=_TASK_TYPE,
        algorithm="lightgbm_lambdamart",
        training_data_hash=hash_dataframe(pd.DataFrame({"query": [q for q, _ in labeled_queries]})),
        hyperparameters={"n_estimators": str(payload.n_estimators)},
        metrics={"num_queries": float(len(labeled_queries))},
        feature_schema=feature_names,
    )

    retriever.set_ranker(ranker)
    request.app.state.active_ranker = ranker

    return RankingTrainResponse(
        trained=True,
        algorithm="lightgbm_lambdamart",
        sample_size=len(labeled_queries),
        feature_importance=[FeatureImportanceOut(feature_name=f.feature_name, importance=f.importance) for f in importance],
    )


@router.get("/status", response_model=RankingStatusOut)
async def ranking_status(request: Request) -> RankingStatusOut:
    active_ranker = getattr(request.app.state, "active_ranker", None)
    if active_ranker is None:
        # No Ranker has ever been injected into the live retriever — final
        # ordering is still the pre-O1 default (sort by cross-encoder score),
        # not Reciprocal Rank Fusion, so the status must say so honestly.
        return RankingStatusOut(active_ranker="cross_encoder_score_default", is_trained=False)
    is_trained = isinstance(active_ranker, LightGBMRanker) and active_ranker.is_trained
    return RankingStatusOut(active_ranker="lightgbm_lambdamart" if is_trained else "reciprocal_rank_fusion_fallback", is_trained=is_trained)
