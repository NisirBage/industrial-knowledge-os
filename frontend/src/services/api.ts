const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

export interface Citation {
  index: number;
  chunk_id: string;
  document_id: string;
  page_number: number;
  source: string;
  entity_ids: string[];
}

export interface Evidence {
  chunk_id: string;
  document_id: string;
  text: string;
  page_number: number;
  entity_ids: string[];
  relation_ids: string[];
  graph_score: number;
  vector_score: number;
  bm25_score: number;
  reranker_score: number;
  source: string;
  confidence: number;
  agent: string;
}

export interface Contradiction {
  conflicting_evidence: string[];
  confidence: number;
  severity: string;
  resolution_strategy: string;
  recommended_clarification: string;
  description: string;
}

export interface Recommendation {
  id: string;
  description: string;
  priority: string;
  severity: string;
  estimated_impact: string;
  required_personnel: string[];
  required_equipment: string[];
  required_documents: string[];
  estimated_downtime: string;
  confidence: number;
  justification: string;
  supporting_evidence: string[];
}

export interface ChatResponse {
  summary: string;
  confidence: number;
  citations: Citation[];
  evidence: Evidence[];
  recommended_actions: string[];
  reasoning_summary: string;
  contradictions: Contradiction[];
  recommendations: Recommendation[];
  latencies: Record<string, number>;
}

export interface DocumentSummary {
  id: string;
  filename: string;
  content_type: string;
  num_chunks: number;
  num_entities: number;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  num_chunks: number;
  num_entities: number;
  num_relations: number;
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function sendChatQuery(query: string): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  return handleResponse<ChatResponse>(response);
}

export async function uploadDocument(file: File): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/documents/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<DocumentUploadResponse>(response);
}

export async function listDocuments(): Promise<DocumentSummary[]> {
  const response = await fetch(`${API_BASE_URL}/documents`);
  return handleResponse<DocumentSummary[]>(response);
}

// --- Research (M5) ---

export interface NamedMetric {
  name: string;
  value: number;
}

export interface Experiment {
  experiment_id: string;
  name: string;
  description: string;
  configuration: Record<string, string>;
  reranker: string;
  ontology_version: string;
  dataset: string;
  timestamp: string;
  status: string;
  metrics: Record<string, number>;
  duration: number | null;
  random_seed: number | null;
  git_commit: string | null;
  error: string | null;
}

export interface BenchmarkReport {
  benchmark_name: string;
  metrics: NamedMetric[];
  duration_seconds: number;
  dataset_name: string;
  sample_size: number;
  timestamp: string;
}

export interface ABTestResult {
  arm_a_name: string;
  arm_b_name: string;
  metric_name: string;
  mean_a: number;
  mean_b: number;
  mean_delta: number;
  t_statistic: number;
  p_value: number;
  winner: string;
}

export interface CalibrationBucket {
  confidence_bucket: string;
  mean_confidence: number;
  actual_accuracy: number | null;
  sample_count: number;
}

export const AUTO_RUN_BENCHMARKS = ["entity_extraction", "relation_extraction", "knowledge_graph"] as const;

export async function listExperiments(): Promise<Experiment[]> {
  const response = await fetch(`${API_BASE_URL}/research/experiments`);
  return handleResponse<Experiment[]>(response);
}

export async function runBenchmark(benchmarkName: string, sampleSize = 20, seed = 42): Promise<BenchmarkReport> {
  const response = await fetch(`${API_BASE_URL}/research/benchmarks/${benchmarkName}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_size: sampleSize, seed }),
  });
  return handleResponse<BenchmarkReport>(response);
}

export async function getBenchmarkHistory(benchmarkName: string): Promise<Experiment[]> {
  const response = await fetch(`${API_BASE_URL}/research/benchmarks/${benchmarkName}/history`);
  return handleResponse<Experiment[]>(response);
}

export async function runEntityExtractionABTest(sampleSize = 20, seed = 42): Promise<ABTestResult> {
  const response = await fetch(`${API_BASE_URL}/research/ab-test/entity-extraction`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_size: sampleSize, seed }),
  });
  return handleResponse<ABTestResult>(response);
}

export async function getCalibration(): Promise<{ buckets: CalibrationBucket[] }> {
  const response = await fetch(`${API_BASE_URL}/research/calibration`);
  return handleResponse<{ buckets: CalibrationBucket[] }>(response);
}

// --- ML Laboratory (M6) ---

export interface FeatureImportance {
  feature_name: string;
  importance: number;
}

export interface CalibrationPoint {
  predicted_probability: number;
  observed_frequency: number;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
}

export interface ModelRecord {
  model_id: string;
  task_type: string;
  algorithm: string;
  training_data_hash: string;
  hyperparameters: Record<string, string>;
  metrics: Record<string, number>;
  feature_schema: string[];
  timestamp: string;
  author: string;
  experiment_id: string | null;
  serialization_path: string;
  serialization_format: string;
}

export interface MLTrainingResult {
  model: ModelRecord;
  feature_importance: FeatureImportance[];
  calibration_curve: CalibrationPoint[];
  confusion_matrix: number[][] | null;
  roc_curve: RocPoint[];
  sample_size: number;
  duration_seconds: number;
}

export interface FeatureDefinition {
  name: string;
  description: string;
  source: string;
  owner: string;
  datatype: string;
  version: string;
  generation_time: string;
}

export interface DriftResult {
  psi: number;
  severity: string;
}

export const TRAINABLE_ML_TASKS = ["failure_prediction", "maintenance_priority"] as const;

export async function trainModel(taskType: string, sampleSize = 300, seed = 42): Promise<MLTrainingResult> {
  const response = await fetch(`${API_BASE_URL}/ml/train/${taskType}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_size: sampleSize, seed }),
  });
  return handleResponse<MLTrainingResult>(response);
}

export async function listModels(taskType?: string): Promise<ModelRecord[]> {
  const url = taskType ? `${API_BASE_URL}/ml/models?task_type=${taskType}` : `${API_BASE_URL}/ml/models`;
  const response = await fetch(url);
  return handleResponse<ModelRecord[]>(response);
}

export async function getFeatureDefinitions(): Promise<FeatureDefinition[]> {
  const response = await fetch(`${API_BASE_URL}/ml/feature-definitions`);
  return handleResponse<FeatureDefinition[]>(response);
}

export async function computeDrift(baseline: number[], recent: number[]): Promise<DriftResult> {
  const response = await fetch(`${API_BASE_URL}/ml/drift`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ baseline, recent }),
  });
  return handleResponse<DriftResult>(response);
}

// --- Vision (M7) ---

export interface VisionDocument {
  id: string;
  filename: string;
  source_type: string;
  content_type: string;
  width: number;
  height: number;
  size_bytes: number;
  equipment_hint: string | null;
}

export interface OCRRegion {
  text: string;
  confidence: number;
  bounding_box: number[];
  graph_node_id: string | null;
}

export interface DiagramSymbol {
  symbol_type: string;
  confidence: number;
  bounding_box: number[];
  graph_node_id: string | null;
}

export interface Detection {
  equipment_type: string;
  confidence: number;
  bounding_box: number[];
  graph_node_id: string | null;
}

export interface GaugeReading {
  needle_angle_degrees: number | null;
  normalized_value: number | null;
  confidence: number;
  valid: boolean;
  validation_note: string;
  bounding_box: number[];
  graph_node_id: string | null;
}

export interface Defect {
  defect_type: string;
  confidence: number;
  severity: string;
  bounding_box: number[];
  graph_node_id: string | null;
}

export interface VisionAnalysis {
  image_id: string;
  ocr_regions: OCRRegion[];
  diagram_symbols: DiagramSymbol[];
  detections: Detection[];
  gauge_readings: GaugeReading[];
  defects: Defect[];
}

export interface MultimodalQueryResponse {
  evidence: Evidence[];
}

export const VISION_SOURCE_TYPES = [
  "inspection_photo",
  "equipment_photo",
  "pid_diagram",
  "electrical_schematic",
  "engineering_drawing",
  "maintenance_image",
  "handwritten_note",
] as const;

export const VISION_BENCHMARKS = [
  "vision_ocr",
  "vision_diagram_parsing",
  "vision_equipment_recognition",
  "vision_gauge_reading",
  "vision_defect_detection",
  "vision_multimodal_retrieval",
  "vision_latency",
] as const;

export async function uploadVisionImage(file: File, sourceType: string, equipmentHint?: string): Promise<VisionDocument> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_type", sourceType);
  if (equipmentHint) formData.append("equipment_hint", equipmentHint);
  const response = await fetch(`${API_BASE_URL}/vision/images`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<VisionDocument>(response);
}

export async function listVisionImages(): Promise<VisionDocument[]> {
  const response = await fetch(`${API_BASE_URL}/vision/images`);
  return handleResponse<VisionDocument[]>(response);
}

export async function getVisionImage(imageId: string): Promise<VisionDocument> {
  const response = await fetch(`${API_BASE_URL}/vision/images/${imageId}`);
  return handleResponse<VisionDocument>(response);
}

export async function analyzeVisionImage(imageId: string): Promise<VisionAnalysis> {
  const response = await fetch(`${API_BASE_URL}/vision/images/${imageId}/analyze`, {
    method: "POST",
  });
  return handleResponse<VisionAnalysis>(response);
}

export async function multimodalQuery(query: string, topK = 8): Promise<MultimodalQueryResponse> {
  const response = await fetch(`${API_BASE_URL}/vision/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  return handleResponse<MultimodalQueryResponse>(response);
}

export function visionImageUrl(imageId: string): string {
  return `${API_BASE_URL}/vision/images/${imageId}/file`;
}

// --- Graph Intelligence (M9) ---

export interface GraphSummary {
  node_count: number;
  edge_count: number;
  node_types: string[];
  relation_types: string[];
}

export interface GraphNode {
  id: string;
  type: string;
  canonical_name: string;
  confidence: number;
  community_id: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation_type: string;
  confidence: number;
}

export interface GraphExport {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface NodeFeatures {
  node_id: string;
  canonical_name: string;
  equipment_type: string;
  degree: number;
  pagerank: number;
  betweenness: number;
  community_id: number | null;
  maintenance_count: number;
  incident_count: number;
  visual_observation_count: number;
  ocr_confidence: number | null;
  defect_count: number;
  prediction_confidence: number | null;
  recommendation_acceptance_rate: number | null;
}

export interface EmbeddingTrainResult {
  model_id: string;
  model_name: string;
  dimensions: number;
  node_count: number;
  training_seconds: number;
}

export interface SimilarNode {
  node_id: string;
  canonical_name: string;
  score: number;
}

export interface GraphExplanation {
  node_id: string;
  nearest_neighbors: SimilarNode[];
  important_edges: string[];
  community_id: number | null;
  supporting_evidence: string[];
  confidence: number;
  text: string;
}

export interface LinkPredictionCandidate {
  source_id: string;
  target_id: string;
  target_canonical_name: string;
  confidence: number;
  embedding_similarity: number;
  explanation: string;
}

export interface NodeClassificationResult {
  registered: boolean;
  task_type: string;
  algorithm: string | null;
  sample_size: number;
  feature_importance: FeatureImportance[];
}

export interface PropagationStep {
  node_id: string;
  canonical_name: string;
  depth: number;
  propagated_risk: number;
  path: string[];
}

export interface PropagationResult {
  source_node_id: string;
  propagation_type: string;
  steps: PropagationStep[];
}

export const GRAPH_EMBEDDING_MODELS = ["node2vec", "deepwalk", "gcn", "graphsage", "gat"] as const;
export const PROPAGATION_TYPES = ["failure", "leak", "maintenance_delay", "inspection_gap", "compliance_violation"] as const;
export const GRAPH_BENCHMARKS = [
  "graph_embedding_node2vec",
  "graph_embedding_deepwalk",
  "graph_embedding_gcn",
  "graph_embedding_graphsage",
  "graph_embedding_gat",
  "graph_link_prediction",
  "graph_node_classification",
] as const;

export async function getGraphSummary(): Promise<GraphSummary> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/summary`);
  return handleResponse<GraphSummary>(response);
}

export async function getGraphExport(): Promise<GraphExport> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/graph`);
  return handleResponse<GraphExport>(response);
}

export async function getNodeFeatures(equipmentName: string): Promise<NodeFeatures> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/features/${encodeURIComponent(equipmentName)}`);
  return handleResponse<NodeFeatures>(response);
}

export async function listGraphEmbeddingModels(): Promise<string[]> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/embeddings/models`);
  return handleResponse<string[]>(response);
}

export async function trainGraphEmbeddings(modelName: string, dimensions = 32): Promise<EmbeddingTrainResult> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/embeddings/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_name: modelName, dimensions }),
  });
  return handleResponse<EmbeddingTrainResult>(response);
}

export async function getSimilarNodes(equipmentName: string, topK = 5): Promise<SimilarNode[]> {
  const response = await fetch(
    `${API_BASE_URL}/graph-intelligence/similarity/${encodeURIComponent(equipmentName)}?top_k=${topK}`,
  );
  return handleResponse<SimilarNode[]>(response);
}

export async function explainNode(equipmentName: string): Promise<GraphExplanation> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/explain/${encodeURIComponent(equipmentName)}`);
  return handleResponse<GraphExplanation>(response);
}

export async function getLinkPredictionCandidates(equipmentName: string, topK = 5): Promise<LinkPredictionCandidate[]> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/link-prediction/candidates`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ equipment_name: equipmentName, top_k: topK }),
  });
  return handleResponse<LinkPredictionCandidate[]>(response);
}

export async function trainNodeClassifier(taskType: string, equipmentNames: string[]): Promise<NodeClassificationResult> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/node-classification/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_type: taskType, equipment_names: equipmentNames }),
  });
  return handleResponse<NodeClassificationResult>(response);
}

export async function simulatePropagation(
  equipmentName: string,
  propagationType = "failure",
  maxDepth = 3,
): Promise<PropagationResult> {
  const response = await fetch(`${API_BASE_URL}/graph-intelligence/propagation/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ equipment_name: equipmentName, propagation_type: propagationType, max_depth: maxDepth }),
  });
  return handleResponse<PropagationResult>(response);
}
