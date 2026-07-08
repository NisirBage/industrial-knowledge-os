"""Graph Dataset data model (M9 Component 1)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    canonical_name: str
    confidence: float
    properties: dict
    # Architecture-only for M9: no current graph relation carries a real
    # timestamp (see TimelineEngine's own docstring on this same gap), so
    # temporal-edge support below has no real data to attach to yet. Node
    # snapshots aren't temporal either, but the field is symmetric with
    # GraphEdge.timestamp for a future point where node state is versioned.
    timestamp: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    relation_type: str
    confidence: float
    # Architecture-only: always None today. A future ingestion pipeline that
    # captures real event dates (maintenance date, incident date) would
    # populate this without changing GraphDatasetBuilder's export logic —
    # every to_*() method already threads this field through.
    timestamp: str | None = None


@dataclass(frozen=True)
class GraphDataset:
    """The builder's own intermediate representation — every to_*() export
    format is derived from this, so exports are consistent with each other
    by construction, not by re-deriving from the graph repository each time.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    # Deterministic node ordering (sorted by id) captured once at build()
    # time — every export (PyG tensor rows, CSV row order, adjacency index)
    # uses this same mapping, so two exports of the same graph state produce
    # byte-identical output.
    node_id_to_index: dict[str, int] = field(default_factory=dict)

    @property
    def node_types(self) -> list[str]:
        return sorted({n.type for n in self.nodes})

    @property
    def relation_types(self) -> list[str]:
        return sorted({e.relation_type for e in self.edges})
