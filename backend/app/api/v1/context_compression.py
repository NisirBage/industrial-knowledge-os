"""Context Compression API (O1.2): retrieves real evidence via the same
HybridRetriever every other endpoint uses, then runs it through
ContextCompressionEngine — a preview/benchmarking surface, not a new
retrieval path (the live chat pipeline doesn't call this; compression stays
optional and additive per this milestone's architectural constraint).
"""

import dataclasses

from fastapi import APIRouter, HTTPException, Request

from app.context_compression.engine import CompressionLevel, ContextCompressionEngine
from app.retrieval.explain import explain_evidence
from app.schemas.chat import EvidenceOut
from app.schemas.context_compression import (
    CompressionRequest,
    CompressionResponseOut,
    CompressionStatsOut,
    EvidenceClusterOut,
)

router = APIRouter(prefix="/context-compression", tags=["context-compression"])


@router.post("/preview", response_model=CompressionResponseOut)
async def preview_compression(payload: CompressionRequest, request: Request) -> CompressionResponseOut:
    try:
        level = CompressionLevel(payload.level)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Unknown level {payload.level!r}. Valid: {[lvl.value for lvl in CompressionLevel]}") from exc

    retriever = request.app.state.retriever
    evidence = retriever.retrieve(payload.query, top_k=payload.top_k)

    engine = ContextCompressionEngine(embedder=request.app.state.embedder)
    bundle = engine.compress(evidence, level=level)

    def _to_evidence_out(item, rank: int) -> EvidenceOut:
        return EvidenceOut(**dataclasses.asdict(item), final_rank=rank, explanation=explain_evidence(item, rank))

    return CompressionResponseOut(
        clusters=[
            EvidenceClusterOut(
                cluster_id=cluster.cluster_id,
                cluster_type=cluster.cluster_type,
                representative=_to_evidence_out(cluster.representative, i + 1),
                member_count=cluster.member_count,
                confidence=cluster.confidence,
            )
            for i, cluster in enumerate(bundle.clusters)
        ],
        compressed_evidence=[_to_evidence_out(item, i + 1) for i, item in enumerate(bundle.compressed_evidence)],
        stats=CompressionStatsOut(
            original_count=bundle.stats.original_count,
            compressed_count=bundle.stats.compressed_count,
            exact_duplicate_count=bundle.stats.exact_duplicate_count,
            near_duplicate_count=bundle.stats.near_duplicate_count,
            cluster_count=bundle.stats.cluster_count,
            compression_ratio=bundle.stats.compression_ratio,
            original_tokens_estimate=bundle.stats.original_tokens_estimate,
            compressed_tokens_estimate=bundle.stats.compressed_tokens_estimate,
            token_reduction_ratio=bundle.stats.token_reduction_ratio,
            summary=bundle.stats.summary,
        ),
    )
