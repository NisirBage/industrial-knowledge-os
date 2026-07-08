from pydantic import BaseModel

from app.schemas.ml import FeatureImportanceOut


class GraphSummaryOut(BaseModel):
    node_count: int
    edge_count: int
    node_types: list[str]
    relation_types: list[str]


class GraphNodeOut(BaseModel):
    id: str
    type: str
    canonical_name: str
    confidence: float
    community_id: int | None = None


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float


class GraphExportOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


class NodeFeaturesOut(BaseModel):
    node_id: str
    canonical_name: str
    equipment_type: str
    degree: int
    pagerank: float
    betweenness: float
    community_id: int
    maintenance_count: int
    incident_count: int
    visual_observation_count: int
    ocr_confidence: float | None
    defect_count: int
    prediction_confidence: float | None
    recommendation_acceptance_rate: float | None


class EmbeddingTrainRequest(BaseModel):
    model_name: str = "node2vec"
    dimensions: int = 32


class EmbeddingTrainResponse(BaseModel):
    model_id: str
    model_name: str
    dimensions: int
    node_count: int
    training_seconds: float


class SimilarNodeOut(BaseModel):
    node_id: str
    canonical_name: str
    score: float


class ExplanationOut(BaseModel):
    node_id: str
    nearest_neighbors: list[SimilarNodeOut]
    important_edges: list[dict]
    community_id: int | None
    supporting_evidence: list[dict]
    confidence: float
    text: str


class LinkPredictionCandidateOut(BaseModel):
    source_id: str
    target_id: str
    target_canonical_name: str
    confidence: float
    embedding_similarity: float
    explanation: str


class LinkPredictionRequest(BaseModel):
    equipment_name: str
    top_k: int = 5


class NodeClassificationTrainRequest(BaseModel):
    task_type: str
    equipment_names: list[str]


class NodeClassificationTrainResponse(BaseModel):
    registered: bool
    task_type: str
    algorithm: str | None = None
    sample_size: int
    feature_importance: list[FeatureImportanceOut] = []


class PropagationRequest(BaseModel):
    equipment_name: str
    propagation_type: str = "failure"
    max_depth: int = 3
    decay_factor: float = 0.5
    initial_risk: float = 1.0


class PropagationStepOut(BaseModel):
    node_id: str
    canonical_name: str
    depth: int
    propagated_risk: float
    path: list[str]


class PropagationResultOut(BaseModel):
    source_node_id: str
    propagation_type: str
    steps: list[PropagationStepOut]
