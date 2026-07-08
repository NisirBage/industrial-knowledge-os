"""Graph Feature Store (M9 Component 2): the graph-native evolution of M6's
FeatureStore — same versioned-metadata discipline (reuses the exact
`FeatureDefinition` dataclass, no parallel metadata model), but every value
now comes from graph structure and multimodal graph content rather than a
per-equipment rule extractor.

Two feature families, computed differently:
  - **Structural** (degree, pagerank, betweenness, community_id): global
    graph algorithms (networkx + python-louvain), computed once per
    `compute_all()` call and shared across every node — recomputing
    pagerank per node would be both wrong (it's not a per-node quantity)
    and wasteful.
  - **Domain** (maintenance/incident/defect counts, OCR confidence,
    prediction confidence, recommendation acceptance): per-node lookups
    against the same GraphRepository and LearningRepository every other
    subsystem already reads from — no new data source, no separate
    tracking table.
"""

import networkx as nx
from community import community_louvain

from app.graph_intelligence.dataset.builder import GraphDatasetBuilder
from app.graph_intelligence.dataset.models import GraphDataset
from app.graph_intelligence.feature_store.models import NodeFeatures
from app.ml.feature_store.models import FeatureDefinition
from app.ontology.hierarchy import descendants_of
from app.ontology.labels import EntityLabel
from app.repositories.graph_repository import GraphRepositoryProtocol
from app.repositories.learning_repository import LearningRepositoryProtocol

_GENERATION_TIME = "runtime"

_FEATURE_DEFINITIONS: list[FeatureDefinition] = [
    FeatureDefinition("degree", "Number of 1-hop graph neighbors", "networkx.degree", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("pagerank", "PageRank centrality over the full graph", "networkx.pagerank", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("betweenness", "Betweenness centrality over the full graph", "networkx.betweenness_centrality", "platform", "float", "v1", _GENERATION_TIME),
    FeatureDefinition("community_id", "Louvain community assignment", "community_louvain.best_partition", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("equipment_type", "Ontology/vision node type", "graph_dataset.node.type", "platform", "str", "v1", _GENERATION_TIME),
    FeatureDefinition("maintenance_count", "Count of maintenance-activity graph neighbors", "graph_repository.maintenance_history", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("incident_count", "Count of incident graph neighbors", "graph_repository.incident_history", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("visual_observation_count", "Count of connected Vision Knowledge Graph nodes (M7)", "graph_repository.find_neighbors", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("ocr_confidence", "Mean confidence of connected OCRRegion nodes", "graph_repository.get_node_properties", "platform", "float | None", "v1", _GENERATION_TIME),
    FeatureDefinition("defect_count", "Count of connected Defect nodes", "graph_repository.find_neighbors", "platform", "int", "v1", _GENERATION_TIME),
    FeatureDefinition("prediction_confidence", "Most recent Prediction Engine confidence for this node", "learning_repository.latest_prediction_confidence", "platform", "float | None", "v1", _GENERATION_TIME),
    FeatureDefinition("recommendation_acceptance_rate", "Historical recommendation acceptance rate for this node", "learning_repository.recommendation_acceptance_rate", "platform", "float | None", "v1", _GENERATION_TIME),
]

_VISUAL_EVIDENCE_TYPES = {label.value for label in descendants_of(EntityLabel.VISUAL_EVIDENCE)}


class GraphFeatureStore:
    def __init__(self, graph_repository: GraphRepositoryProtocol, learning_repository: LearningRepositoryProtocol):
        self._graph_repository = graph_repository
        self._learning_repository = learning_repository
        self._dataset_builder = GraphDatasetBuilder(graph_repository)

    def feature_definitions(self) -> list[FeatureDefinition]:
        return list(_FEATURE_DEFINITIONS)

    def structural_features(self, dataset: GraphDataset) -> dict[str, dict]:
        graph = self._dataset_builder.to_networkx(dataset).to_undirected()
        if graph.number_of_nodes() == 0:
            return {}

        degree = dict(graph.degree())
        pagerank = nx.pagerank(graph) if graph.number_of_edges() > 0 else {n: 0.0 for n in graph.nodes()}
        betweenness = nx.betweenness_centrality(graph)
        community = community_louvain.best_partition(graph) if graph.number_of_edges() > 0 else {n: 0 for n in graph.nodes()}

        return {
            node_id: {
                "degree": degree.get(node_id, 0),
                "pagerank": round(pagerank.get(node_id, 0.0), 6),
                "betweenness": round(betweenness.get(node_id, 0.0), 6),
                "community_id": community.get(node_id, 0),
            }
            for node_id in graph.nodes()
        }

    async def compute_node_features(self, node_id: str, canonical_name: str, node_type: str, structural: dict) -> NodeFeatures:
        maintenance_count = len(self._graph_repository.maintenance_history(canonical_name))
        incident_count = len(self._graph_repository.incident_history(canonical_name))

        neighbors = self._graph_repository.find_neighbors(node_id, depth=1)
        visual_neighbors = [n for n in neighbors if n["type"] in _VISUAL_EVIDENCE_TYPES]
        defect_count = sum(1 for n in visual_neighbors if n["type"] == EntityLabel.DEFECT.value)

        ocr_neighbors = [n for n in visual_neighbors if n["type"] == EntityLabel.OCR_REGION.value]
        ocr_confidences = [self._graph_repository.get_node_properties(n["id"]).get("confidence") for n in ocr_neighbors]
        ocr_confidences = [c for c in ocr_confidences if c is not None]
        ocr_confidence = round(sum(ocr_confidences) / len(ocr_confidences), 4) if ocr_confidences else None

        prediction_confidence = await self._learning_repository.latest_prediction_confidence(canonical_name)
        recommendation_acceptance_rate = await self._learning_repository.recommendation_acceptance_rate(canonical_name)

        return NodeFeatures(
            node_id=node_id,
            canonical_name=canonical_name,
            equipment_type=node_type,
            degree=structural.get("degree", 0),
            pagerank=structural.get("pagerank", 0.0),
            betweenness=structural.get("betweenness", 0.0),
            community_id=structural.get("community_id", 0),
            maintenance_count=maintenance_count,
            incident_count=incident_count,
            visual_observation_count=len(visual_neighbors),
            ocr_confidence=ocr_confidence,
            defect_count=defect_count,
            prediction_confidence=prediction_confidence,
            recommendation_acceptance_rate=recommendation_acceptance_rate,
        )

    async def compute_all(self, dataset: GraphDataset | None = None) -> list[NodeFeatures]:
        dataset = dataset or self._dataset_builder.build()
        structural_by_node = self.structural_features(dataset)
        return [
            await self.compute_node_features(node.id, node.canonical_name, node.type, structural_by_node.get(node.id, {}))
            for node in dataset.nodes
        ]
