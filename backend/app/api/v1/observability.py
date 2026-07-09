from fastapi import APIRouter, Request

from app.repositories.learning_repository import LearningRepositoryProtocol
from app.schemas.observability import CacheStatsOut, ObservabilityMetricsOut

router = APIRouter(tags=["observability"])


@router.get("/observability/metrics", response_model=ObservabilityMetricsOut)
async def get_observability_metrics(request: Request) -> ObservabilityMetricsOut:
    learning_repository: LearningRepositoryProtocol = request.app.state.learning_repository

    return ObservabilityMetricsOut(
        feedback_volume=await learning_repository.feedback_volume(),
        average_prediction_confidence=await learning_repository.average_prediction_confidence(),
        average_prediction_latency_seconds=await learning_repository.average_model_accuracy("prediction_latency_seconds"),
        average_feature_generation_latency_seconds=await learning_repository.average_model_accuracy("feature_generation_latency_seconds"),
    )


@router.get("/observability/cache-stats", response_model=list[CacheStatsOut])
async def get_cache_stats(request: Request) -> list[CacheStatsOut]:
    """O1 Component 6: real hit/miss/eviction counters for every named
    retrieval cache — not a synthetic/estimated figure, the same `CacheStats`
    each `LRUTTLCache` instance tracks internally."""
    named_caches = {
        "retrieval_embedding": request.app.state.retrieval_embedding_cache,
        "retrieval_expansion": request.app.state.retrieval_expansion_cache,
        "retrieval_reranker": request.app.state.retrieval_reranker_cache,
        "retrieval_result": request.app.state.retrieval_result_cache,
    }
    return [
        CacheStatsOut(
            name=name,
            hits=cache.stats.hits,
            misses=cache.stats.misses,
            evictions=cache.stats.evictions,
            size=cache.stats.size,
            max_size=cache.stats.max_size,
            hit_ratio=cache.stats.hit_ratio,
        )
        for name, cache in named_caches.items()
    ]
