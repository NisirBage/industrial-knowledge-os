"""Entity Resolution API (O1.2): find likely-duplicate graph nodes, merge
one into another, and roll a merge back. `app.state.merge_history` is a
simple in-process dict of merge_id -> MergeRecord — real provenance for the
lifetime of the process, not a persisted audit table (a documented scope
choice, not a hidden gap: see docs/architecture/retrieval-optimization.md).
"""

from fastapi import APIRouter, HTTPException, Request

from app.entity_resolution.merger import GraphMerger
from app.entity_resolution.resolver import EntityResolver, KnownEntity
from app.ontology.labels import EntityLabel
from app.schemas.entity_resolution import (
    DuplicateCandidateOut,
    MergeRequest,
    MergeResponseOut,
    RollbackRequest,
    RollbackResponseOut,
)

router = APIRouter(prefix="/entity-resolution", tags=["entity-resolution"])


@router.get("/duplicates", response_model=list[DuplicateCandidateOut])
async def find_duplicates(request: Request, threshold: float = 0.82, limit: int = 200) -> list[DuplicateCandidateOut]:
    graph_repository = request.app.state.graph_repository
    nodes = graph_repository.all_nodes()[:limit]
    resolver = EntityResolver(merge_threshold=threshold)

    known_entities = [KnownEntity(node_id=n["id"], canonical_name=n["canonical_name"], label=EntityLabel(n["type"])) for n in nodes]

    candidates: list[DuplicateCandidateOut] = []
    seen_pairs: set[frozenset[str]] = set()
    for i, node in enumerate(nodes):
        others = known_entities[:i] + known_entities[i + 1 :]
        result = resolver.resolve(node["canonical_name"], EntityLabel(node["type"]), others)
        if result is None:
            continue
        pair_key = frozenset({node["id"], result.canonical_id})
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        candidates.append(
            DuplicateCandidateOut(
                source_id=node["id"],
                source_name=node["canonical_name"],
                target_id=result.canonical_id,
                target_name=result.canonical_name,
                confidence=result.confidence,
                merge_reason=result.merge_reason,
            )
        )

    return candidates


@router.post("/merge", response_model=MergeResponseOut)
async def merge_nodes(payload: MergeRequest, request: Request) -> MergeResponseOut:
    graph_repository = request.app.state.graph_repository
    merger = GraphMerger(graph_repository)
    try:
        record = merger.merge(payload.source_id, payload.target_id, payload.reason)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    request.app.state.merge_history[record.merge_id] = record
    return MergeResponseOut(
        merge_id=record.merge_id, source_id=payload.source_id, target_id=record.target_id, reason=record.reason, timestamp=record.timestamp
    )


@router.post("/rollback", response_model=RollbackResponseOut)
async def rollback_merge(payload: RollbackRequest, request: Request) -> RollbackResponseOut:
    merge_history: dict = request.app.state.merge_history
    record = merge_history.get(payload.merge_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No merge record with id {payload.merge_id!r}")

    graph_repository = request.app.state.graph_repository
    merger = GraphMerger(graph_repository)
    restored_id = merger.rollback(record)
    del merge_history[payload.merge_id]
    return RollbackResponseOut(restored_node_id=restored_id)
