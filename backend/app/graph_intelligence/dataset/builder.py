"""Graph Dataset Builder (M9 Component 1): exports the existing Knowledge
Graph (GraphRepositoryProtocol) into every downstream format the Embedding
Framework and external tooling need — PyTorch Geometric (homogeneous and
heterogeneous), NetworkX, CSV, GraphML, JSON. One `build()` call produces a
deterministic `GraphDataset`; every `to_*()` export is derived from that same
object, so two exports of one graph snapshot are always consistent with each
other.

Determinism: nodes are sorted by id before indexing, so `node_id_to_index`
(and therefore every tensor row / CSV row / adjacency position derived from
it) is stable across repeated calls against the same graph state — the
explicit requirement from the M9 spec ("Dataset creation must be
deterministic").

Multimodal node types are automatic, not a special case: `all_nodes()`
already returns every node the graph has, whether it came from text
ingestion (Pump, Valve, Incident...) or the Vision Knowledge Graph (Image,
OCRRegion, DetectedObject, Defect, GaugeReading, DiagramSymbol...) — M7's
vision nodes and M1-M6's text-derived nodes are exported side by side with
no extra code.
"""

import csv
import json
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData

from app.graph_intelligence.dataset.models import GraphDataset, GraphEdge, GraphNode
from app.repositories.graph_repository import GraphRepositoryProtocol


class GraphDatasetBuilder:
    def __init__(self, graph_repository: GraphRepositoryProtocol):
        self._graph_repository = graph_repository

    def build(self) -> GraphDataset:
        raw_nodes = sorted(self._graph_repository.all_nodes(), key=lambda n: n["id"])
        raw_edges = self._graph_repository.all_edges()

        nodes = [
            GraphNode(id=n["id"], type=n["type"], canonical_name=n["canonical_name"], confidence=n["confidence"], properties=n["properties"])
            for n in raw_nodes
        ]
        edges = [
            GraphEdge(source=e["source"], target=e["target"], relation_type=e["relation_type"], confidence=e["confidence"])
            for e in raw_edges
        ]
        node_id_to_index = {node.id: index for index, node in enumerate(nodes)}
        return GraphDataset(nodes=nodes, edges=edges, node_id_to_index=node_id_to_index)

    @staticmethod
    def to_networkx(dataset: GraphDataset) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for node in dataset.nodes:
            # `properties` is nested under its own key rather than splatted
            # as top-level kwargs — a vision node's properties dict (M7)
            # commonly has its own "confidence" key (see
            # VisionKnowledgeGraphBuilder._node_properties()), which would
            # otherwise collide with the node's first-class `confidence`.
            graph.add_node(node.id, type=node.type, canonical_name=node.canonical_name, confidence=node.confidence, properties=node.properties)
        for edge in dataset.edges:
            graph.add_edge(edge.source, edge.target, relation_type=edge.relation_type, confidence=edge.confidence)
        return graph

    @staticmethod
    def _type_one_hot_matrix(dataset: GraphDataset) -> torch.Tensor:
        """Default node feature when no richer Graph Feature Store vector is
        supplied — a one-hot of the node's ontology/vision type. Real,
        structural, deterministic; not a placeholder of zeros. The Feature
        Store (M9 Component 2) supplies a richer `x` for training; this
        keeps the Dataset Builder usable standalone, satisfying "reusable
        dataset builder" independent of Component 2 being wired up yet.
        """
        types = dataset.node_types
        type_index = {t: i for i, t in enumerate(types)}
        matrix = torch.zeros((len(dataset.nodes), len(types)))
        for row, node in enumerate(dataset.nodes):
            matrix[row, type_index[node.type]] = 1.0
        return matrix

    @staticmethod
    def to_pyg_homogeneous(dataset: GraphDataset, node_features: torch.Tensor | None = None) -> Data:
        """Collapses every node type into one homogeneous graph — the shape
        GCN/GraphSAGE/GAT (Component 3) train on by default. `node_features`
        lets a caller (e.g. GraphFeatureStore) supply a richer `x` than the
        built-in type one-hot.
        """
        x = node_features if node_features is not None else GraphDatasetBuilder._type_one_hot_matrix(dataset)
        if not dataset.edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                [[dataset.node_id_to_index[e.source] for e in dataset.edges], [dataset.node_id_to_index[e.target] for e in dataset.edges]],
                dtype=torch.long,
            )
        return Data(x=x, edge_index=edge_index, num_nodes=len(dataset.nodes))

    @staticmethod
    def to_pyg_heterogeneous(dataset: GraphDataset) -> HeteroData:
        """One node-feature block per ontology/vision type and one edge
        block per (source_type, relation_type, target_type) triple — the
        real heterogeneous-graph shape a future heterogeneous GNN (the M9
        spec's explicit "Future heterogeneous GNNs" line) would consume.
        """
        data = HeteroData()
        nodes_by_type: dict[str, list[GraphNode]] = {}
        for node in dataset.nodes:
            nodes_by_type.setdefault(node.type, []).append(node)

        local_index: dict[str, int] = {}
        for node_type, nodes in nodes_by_type.items():
            data[node_type].x = torch.eye(len(nodes)) if nodes else torch.zeros((0, 0))
            for local_i, node in enumerate(nodes):
                local_index[node.id] = local_i

        edges_by_triple: dict[tuple[str, str, str], list[GraphEdge]] = {}
        node_type_by_id = {node.id: node.type for node in dataset.nodes}
        for edge in dataset.edges:
            source_type = node_type_by_id.get(edge.source)
            target_type = node_type_by_id.get(edge.target)
            if source_type is None or target_type is None:
                continue
            edges_by_triple.setdefault((source_type, edge.relation_type, target_type), []).append(edge)

        for (source_type, relation_type, target_type), edges in edges_by_triple.items():
            data[source_type, relation_type, target_type].edge_index = torch.tensor(
                [[local_index[e.source] for e in edges], [local_index[e.target] for e in edges]], dtype=torch.long
            )
        return data

    @staticmethod
    def to_csv(dataset: GraphDataset, nodes_path: Path, edges_path: Path) -> None:
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nodes_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "type", "canonical_name", "confidence"])
            for node in dataset.nodes:
                writer.writerow([node.id, node.type, node.canonical_name, node.confidence])

        with open(edges_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "target", "relation_type", "confidence"])
            for edge in dataset.edges:
                writer.writerow([edge.source, edge.target, edge.relation_type, edge.confidence])

    @staticmethod
    def to_graphml(dataset: GraphDataset, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        graph = nx.MultiDiGraph()
        for node in dataset.nodes:
            graph.add_node(node.id, type=node.type, canonical_name=node.canonical_name, confidence=node.confidence)
        for edge in dataset.edges:
            graph.add_edge(edge.source, edge.target, relation_type=edge.relation_type, confidence=edge.confidence)
        nx.write_graphml(graph, path)

    @staticmethod
    def to_json(dataset: GraphDataset, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": [
                {"id": n.id, "type": n.type, "canonical_name": n.canonical_name, "confidence": n.confidence, "properties": n.properties}
                for n in dataset.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "relation_type": e.relation_type, "confidence": e.confidence}
                for e in dataset.edges
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
