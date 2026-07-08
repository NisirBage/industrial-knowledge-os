import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  GRAPH_BENCHMARKS,
  GRAPH_EMBEDDING_MODELS,
  PROPAGATION_TYPES,
  explainNode,
  getGraphExport,
  getGraphSummary,
  getLinkPredictionCandidates,
  getNodeFeatures,
  getSimilarNodes,
  listGraphEmbeddingModels,
  simulatePropagation,
  trainGraphEmbeddings,
  trainNodeClassifier,
} from "./api";
import type { GraphExplanation, GraphExport, GraphSummary, NodeFeatures } from "./api";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response;
}

const SAMPLE_SUMMARY: GraphSummary = {
  node_count: 6,
  edge_count: 6,
  node_types: ["Pump", "Valve"],
  relation_types: ["FAILED_DUE_TO"],
};

const SAMPLE_EXPORT: GraphExport = {
  nodes: [{ id: "n1", type: "Pump", canonical_name: "P-101", confidence: 0.9, community_id: 0 }],
  edges: [{ source: "n1", target: "n2", relation_type: "CONNECTED_TO", confidence: 0.8 }],
};

const SAMPLE_FEATURES: NodeFeatures = {
  node_id: "n1",
  canonical_name: "P-101",
  equipment_type: "Pump",
  degree: 2,
  pagerank: 0.15,
  betweenness: 0.05,
  community_id: 0,
  maintenance_count: 3,
  incident_count: 1,
  visual_observation_count: 0,
  ocr_confidence: null,
  defect_count: 0,
  prediction_confidence: 0.7,
  recommendation_acceptance_rate: null,
};

const SAMPLE_EXPLANATION: GraphExplanation = {
  node_id: "n1",
  nearest_neighbors: [{ node_id: "n2", canonical_name: "V-101", score: 0.9 }],
  important_edges: ["CONNECTED_TO->n2"],
  community_id: 0,
  supporting_evidence: ["2 connected graph node(s)"],
  confidence: 0.8,
  text: "Nearest embedding neighbors: n2 (0.9). Confidence: 80.0%.",
};

describe("Graph Intelligence API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("getGraphSummary", () => {
    it("fetches the summary", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_SUMMARY));

      const result = await getGraphSummary();

      expect(result).toEqual(SAMPLE_SUMMARY);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/graph-intelligence/summary");
    });

    it("propagates errors from a failed request", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "server error" }, false, 500));

      await expect(getGraphSummary()).rejects.toThrow(/500/);
    });
  });

  describe("getGraphExport", () => {
    it("fetches the full node-link export", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_EXPORT));

      const result = await getGraphExport();

      expect(result).toEqual(SAMPLE_EXPORT);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/graph-intelligence/graph");
    });
  });

  describe("getNodeFeatures", () => {
    it("fetches features for an equipment name", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_FEATURES));

      const result = await getNodeFeatures("P-101");

      expect(result).toEqual(SAMPLE_FEATURES);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/graph-intelligence/features/P-101");
    });

    it("throws a 404 error for unknown equipment", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "not found" }, false, 404));

      await expect(getNodeFeatures("missing")).rejects.toThrow(/404/);
    });
  });

  describe("listGraphEmbeddingModels", () => {
    it("returns the list of available model names", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(["node2vec", "deepwalk", "gcn", "graphsage", "gat"]));

      const result = await listGraphEmbeddingModels();

      expect(result).toEqual(["node2vec", "deepwalk", "gcn", "graphsage", "gat"]);
    });
  });

  describe("trainGraphEmbeddings", () => {
    it("posts model_name and dimensions", async () => {
      vi.mocked(fetch).mockResolvedValue(
        jsonResponse({ model_id: "m1", model_name: "node2vec", dimensions: 8, node_count: 6, training_seconds: 0.25 }),
      );

      const result = await trainGraphEmbeddings("node2vec", 8);

      expect(result.model_id).toBe("m1");
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toContain("/graph-intelligence/embeddings/train");
      expect(init?.method).toBe("POST");
      const body = JSON.parse(init?.body as string);
      expect(body).toEqual({ model_name: "node2vec", dimensions: 8 });
    });

    it("throws a 422 error for an unknown model name", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "unknown model" }, false, 422));

      await expect(trainGraphEmbeddings("not_a_real_model")).rejects.toThrow(/422/);
    });
  });

  describe("getSimilarNodes", () => {
    it("fetches ranked neighbors for an equipment name", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse([{ node_id: "n2", canonical_name: "V-101", score: 0.9 }]));

      const result = await getSimilarNodes("P-101", 3);

      expect(result).toEqual([{ node_id: "n2", canonical_name: "V-101", score: 0.9 }]);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/graph-intelligence/similarity/P-101?top_k=3");
    });

    it("throws a 400 error when no embedding has been trained", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "no embedding" }, false, 400));

      await expect(getSimilarNodes("P-101")).rejects.toThrow(/400/);
    });
  });

  describe("explainNode", () => {
    it("fetches the explanation for an equipment name", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_EXPLANATION));

      const result = await explainNode("P-101");

      expect(result).toEqual(SAMPLE_EXPLANATION);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/graph-intelligence/explain/P-101");
    });
  });

  describe("getLinkPredictionCandidates", () => {
    it("posts equipment_name and top_k and returns ranked candidates", async () => {
      const candidates = [
        {
          source_id: "n1",
          target_id: "n3",
          target_canonical_name: "M-101",
          confidence: 0.6,
          embedding_similarity: 0.7,
          explanation: "likely link",
        },
      ];
      vi.mocked(fetch).mockResolvedValue(jsonResponse(candidates));

      const result = await getLinkPredictionCandidates("P-101", 3);

      expect(result).toEqual(candidates);
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toContain("/graph-intelligence/link-prediction/candidates");
      const body = JSON.parse(init?.body as string);
      expect(body).toEqual({ equipment_name: "P-101", top_k: 3 });
    });
  });

  describe("trainNodeClassifier", () => {
    it("posts task_type and equipment_names", async () => {
      vi.mocked(fetch).mockResolvedValue(
        jsonResponse({ registered: true, task_type: "failure_risk", algorithm: "graph_RandomForestRegressor", sample_size: 3, feature_importance: [] }),
      );

      const result = await trainNodeClassifier("failure_risk", ["P-101", "V-102"]);

      expect(result.registered).toBe(true);
      const [, init] = vi.mocked(fetch).mock.calls[0];
      const body = JSON.parse(init?.body as string);
      expect(body).toEqual({ task_type: "failure_risk", equipment_names: ["P-101", "V-102"] });
    });

    it("throws a 422 error for an unknown task_type", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "unknown task" }, false, 422));

      await expect(trainNodeClassifier("not_a_real_task", ["P-101"])).rejects.toThrow(/422/);
    });
  });

  describe("simulatePropagation", () => {
    it("posts equipment_name, propagation_type, and max_depth", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ source_node_id: "n1", propagation_type: "failure", steps: [] }));

      await simulatePropagation("P-101", "leak", 2);

      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toContain("/graph-intelligence/propagation/simulate");
      const body = JSON.parse(init?.body as string);
      expect(body).toEqual({ equipment_name: "P-101", propagation_type: "leak", max_depth: 2 });
    });

    it("throws a 404 error for unknown equipment", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "not found" }, false, 404));

      await expect(simulatePropagation("missing")).rejects.toThrow(/404/);
    });
  });

  describe("GRAPH_EMBEDDING_MODELS", () => {
    it("includes every backend embedding model name", () => {
      expect(GRAPH_EMBEDDING_MODELS).toEqual(["node2vec", "deepwalk", "gcn", "graphsage", "gat"]);
    });
  });

  describe("PROPAGATION_TYPES", () => {
    it("includes every backend propagation type", () => {
      expect(PROPAGATION_TYPES).toEqual(["failure", "leak", "maintenance_delay", "inspection_gap", "compliance_violation"]);
    });
  });

  describe("GRAPH_BENCHMARKS", () => {
    it("includes every graph benchmark name the research API exposes", () => {
      expect(GRAPH_BENCHMARKS).toEqual([
        "graph_embedding_node2vec",
        "graph_embedding_deepwalk",
        "graph_embedding_gcn",
        "graph_embedding_graphsage",
        "graph_embedding_gat",
        "graph_link_prediction",
        "graph_node_classification",
      ]);
    });
  });
});
