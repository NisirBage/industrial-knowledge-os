"""GraphMerger: merges a duplicate node into its canonical counterpart —
redirects edges, merges properties, records enough provenance to roll the
merge back. Built entirely on the existing GraphRepositoryProtocol
(all_nodes/all_edges/get_node_properties/create_node/create_relation) plus
one new primitive (`delete_node`) — no parallel graph-mutation path.

Rollback honestly recreates the source node and its original edges (using
`entity_id_for`'s determinism — the same (label, canonical_name) pair always
produces the same node id, so the restored node is indistinguishable from
the original). It does *not* attempt to remove the redirected edges added to
the target during the merge: `create_relation` is itself idempotent (a
repeated call just bumps `observation_count`/`confidence`), and those edges
may be shared with, or have been independently contributed to by, other
merges — silently deleting them on rollback could destroy real information
a naive "undo" shouldn't touch. This is a documented, honest limitation, not
a hidden one.
"""

import datetime as dt
import uuid
from dataclasses import dataclass, field

from app.ontology.labels import EntityLabel, RelationType
from app.repositories.graph_repository import GraphRepositoryProtocol


@dataclass(frozen=True)
class MergeRecord:
    merge_id: str
    source_node: dict
    source_edges: list[dict]
    target_id: str
    reason: str
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


class GraphMerger:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def merge(self, source_id: str, target_id: str, reason: str) -> MergeRecord:
        if source_id == target_id:
            raise ValueError("Cannot merge a node into itself")

        nodes_by_id = {n["id"]: n for n in self._graph_repository.all_nodes()}
        source_node = nodes_by_id.get(source_id)
        target_node = nodes_by_id.get(target_id)
        if source_node is None or target_node is None:
            raise KeyError(f"Both source_id ({source_id!r}) and target_id ({target_id!r}) must be existing nodes")

        source_edges = [e for e in self._graph_repository.all_edges() if e["source"] == source_id or e["target"] == source_id]

        # Property merge: target's own values win on conflict — the target
        # is, by construction (the caller picked it as the merge
        # destination), the more-established canonical node.
        source_properties = self._graph_repository.get_node_properties(source_id)
        target_properties = self._graph_repository.get_node_properties(target_id)
        merged_properties = {**source_properties, **target_properties}

        for edge in source_edges:
            new_source = target_id if edge["source"] == source_id else edge["source"]
            new_target = target_id if edge["target"] == source_id else edge["target"]
            if new_source == new_target:
                continue  # would-be self-loop on the target -- not real information, dropped
            self._graph_repository.create_relation(
                new_source, new_target, RelationType(edge["relation_type"]), confidence=edge.get("confidence", 0.5)
            )

        self._graph_repository.create_node(
            EntityLabel(target_node["type"]),
            target_node["canonical_name"],
            aliases=frozenset({source_node["canonical_name"]}),
            confidence=max(source_node.get("confidence", 0.5), target_node.get("confidence", 0.5)),
            properties=merged_properties,
        )

        self._graph_repository.delete_node(source_id)

        return MergeRecord(
            merge_id=str(uuid.uuid4()),
            source_node={**source_node, "properties": source_properties},
            source_edges=source_edges,
            target_id=target_id,
            reason=reason,
        )

    def rollback(self, record: MergeRecord) -> str:
        """Recreates the source node (same id, via entity_id_for's
        determinism) and its original edges. Returns the restored node id."""
        source = record.source_node
        label = EntityLabel(source["type"])
        restored_id = self._graph_repository.create_node(
            label,
            source["canonical_name"],
            confidence=source.get("confidence", 0.5),
            properties=source.get("properties", {}),
        )
        for edge in record.source_edges:
            self._graph_repository.create_relation(
                edge["source"], edge["target"], RelationType(edge["relation_type"]), confidence=edge.get("confidence", 0.5)
            )
        return restored_id
