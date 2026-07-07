"""Evidence Graph (M3 Component 3) — NOT the Knowledge Graph.

The Knowledge Graph (app/repositories/graph_repository.py) stores durable
industrial knowledge across all documents. The Evidence Graph is built fresh
for a single query from that query's EvidenceBundle and exists only for the
duration of the reasoning session — it is the scaffold the Industrial
Reasoning Graph (app/reasoning) is built on top of, and is never persisted.
"""

from dataclasses import dataclass
from enum import Enum

from app.evidence.models import EvidenceBundle


class EvidenceNodeType(str, Enum):
    QUESTION = "question"
    EVIDENCE = "evidence"
    ENTITY = "entity"
    RELATIONSHIP = "relationship"
    RECOMMENDATION = "recommendation"
    ANSWER = "answer"


@dataclass(frozen=True)
class EvidenceGraphNode:
    id: str
    node_type: EvidenceNodeType
    description: str
    source: str
    confidence: float
    importance: float
    timestamp: str | None = None
    document_id: str = ""
    page_number: int = 0
    chunk_id: str = ""
    entity_id: str | None = None
    relationship_id: str | None = None


@dataclass(frozen=True)
class EvidenceGraphEdge:
    source_id: str
    target_id: str
    label: str


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]

    def add_node(self, node: EvidenceGraphNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: EvidenceGraphEdge) -> None:
        self.edges.append(edge)


_QUESTION_NODE_ID = "question"


class EvidenceGraphBuilder:
    def build(self, query: str, bundle: EvidenceBundle) -> EvidenceGraph:
        graph = EvidenceGraph(nodes=[], edges=[])
        graph.add_node(
            EvidenceGraphNode(
                id=_QUESTION_NODE_ID,
                node_type=EvidenceNodeType.QUESTION,
                description=query,
                source="user",
                confidence=1.0,
                importance=1.0,
            )
        )

        for item in bundle.evidence:
            node_id = f"evidence:{item.chunk_id}"
            graph.add_node(
                EvidenceGraphNode(
                    id=node_id,
                    node_type=EvidenceNodeType.EVIDENCE,
                    description=item.text[:300],
                    source=item.source,
                    confidence=item.confidence,
                    importance=item.confidence,
                    document_id=item.document_id,
                    page_number=item.page_number,
                    chunk_id=item.chunk_id,
                )
            )
            graph.add_edge(EvidenceGraphEdge(source_id=_QUESTION_NODE_ID, target_id=node_id, label="supports"))

        known_entities: set[str] = set()
        for item in bundle.graph_evidence:
            relationship_node_id = f"relationship:{item.chunk_id}"
            graph.add_node(
                EvidenceGraphNode(
                    id=relationship_node_id,
                    node_type=EvidenceNodeType.RELATIONSHIP,
                    description=item.text,
                    source=item.source,
                    confidence=item.confidence,
                    importance=item.confidence,
                    relationship_id=item.chunk_id,
                )
            )
            graph.add_edge(EvidenceGraphEdge(source_id=_QUESTION_NODE_ID, target_id=relationship_node_id, label="relates"))

            for entity_id in item.entity_ids:
                entity_node_id = f"entity:{entity_id}"
                if entity_id not in known_entities:
                    known_entities.add(entity_id)
                    graph.add_node(
                        EvidenceGraphNode(
                            id=entity_node_id,
                            node_type=EvidenceNodeType.ENTITY,
                            description=entity_id,
                            source=item.source,
                            confidence=item.confidence,
                            importance=item.confidence,
                            entity_id=entity_id,
                        )
                    )
                graph.add_edge(EvidenceGraphEdge(source_id=relationship_node_id, target_id=entity_node_id, label="involves"))

        return graph
