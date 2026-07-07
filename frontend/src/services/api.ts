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
