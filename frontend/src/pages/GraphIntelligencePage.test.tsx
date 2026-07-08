import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GraphIntelligencePage } from "./GraphIntelligencePage";
import * as api from "../services/api";
import type {
  EmbeddingTrainResult,
  GraphExplanation,
  GraphExport,
  GraphSummary,
  LinkPredictionCandidate,
  ModelRecord,
  NodeClassificationResult,
  NodeFeatures,
  PropagationResult,
  SimilarNode,
} from "../services/api";

// cytoscape renders to a <canvas> 2D context that jsdom does not implement
// (getContext('2d') returns null without the optional native `canvas`
// package) — mocked here so tests exercise the page's real data flow and
// conditional rendering without depending on canvas support.
vi.mock("cytoscape", () => {
  const instance = {
    on: vi.fn(),
    destroy: vi.fn(),
    nodes: vi.fn(() => ({ removeClass: vi.fn() })),
    getElementById: vi.fn(() => ({ addClass: vi.fn() })),
    zoom: vi.fn(() => 1),
    fit: vi.fn(),
  };
  return { default: vi.fn(() => instance) };
});

vi.mock("../services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/api")>();
  return {
    ...actual,
    getGraphSummary: vi.fn(),
    getGraphExport: vi.fn(),
    getNodeFeatures: vi.fn(),
    trainGraphEmbeddings: vi.fn(),
    getSimilarNodes: vi.fn(),
    explainNode: vi.fn(),
    getLinkPredictionCandidates: vi.fn(),
    trainNodeClassifier: vi.fn(),
    simulatePropagation: vi.fn(),
    listModels: vi.fn(),
    runBenchmark: vi.fn(),
  };
});

const getGraphSummary = vi.mocked(api.getGraphSummary);
const getGraphExport = vi.mocked(api.getGraphExport);
const getNodeFeatures = vi.mocked(api.getNodeFeatures);
const trainGraphEmbeddings = vi.mocked(api.trainGraphEmbeddings);
const getSimilarNodes = vi.mocked(api.getSimilarNodes);
const explainNode = vi.mocked(api.explainNode);
const getLinkPredictionCandidates = vi.mocked(api.getLinkPredictionCandidates);
const trainNodeClassifier = vi.mocked(api.trainNodeClassifier);
const simulatePropagation = vi.mocked(api.simulatePropagation);
const listModels = vi.mocked(api.listModels);
const runBenchmark = vi.mocked(api.runBenchmark);

const SUMMARY: GraphSummary = {
  node_count: 6,
  edge_count: 6,
  node_types: ["Pump", "Valve"],
  relation_types: ["FAILED_DUE_TO"],
};

const GRAPH_EXPORT: GraphExport = {
  nodes: [
    { id: "n1", type: "Pump", canonical_name: "P-101", confidence: 0.9, community_id: 0 },
    { id: "n2", type: "Valve", canonical_name: "V-102", confidence: 0.8, community_id: 0 },
  ],
  edges: [{ source: "n1", target: "n2", relation_type: "FAILED_DUE_TO", confidence: 0.7 }],
};

beforeEach(() => {
  getGraphSummary.mockResolvedValue(SUMMARY);
  getGraphExport.mockResolvedValue(GRAPH_EXPORT);
  getNodeFeatures.mockReset();
  trainGraphEmbeddings.mockReset();
  getSimilarNodes.mockReset();
  explainNode.mockReset();
  getLinkPredictionCandidates.mockReset();
  trainNodeClassifier.mockReset();
  simulatePropagation.mockReset();
  listModels.mockReset();
  runBenchmark.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("GraphIntelligencePage — initial rendering", () => {
  it("renders every major section heading", async () => {
    render(<GraphIntelligencePage />);

    for (const heading of [
      "Filters",
      "Knowledge Graph",
      "Train Embeddings",
      "Link Prediction",
      "Risk Propagation",
      "Node Classification",
      "Graph Research Benchmarks",
    ]) {
      expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
    }
  });

  it("fetches the graph summary and export on mount", async () => {
    render(<GraphIntelligencePage />);

    await waitFor(() => expect(getGraphSummary).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(getGraphExport).toHaveBeenCalledTimes(1));
  });

  it("shows the node and edge counts from the summary", async () => {
    render(<GraphIntelligencePage />);

    expect(await screen.findAllByText("6")).toHaveLength(2);
    expect(await screen.findByText(/Pump, Valve/)).toBeInTheDocument();
  });

  it("renders a checkbox for every node type", async () => {
    render(<GraphIntelligencePage />);

    expect(await screen.findByRole("checkbox", { name: /Pump/ })).toBeInTheDocument();
    expect(await screen.findByRole("checkbox", { name: /Valve/ })).toBeInTheDocument();
  });

  it("shows an error banner when the initial summary fetch fails", async () => {
    getGraphSummary.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<GraphIntelligencePage />);

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });

  it("renders every GRAPH_EMBEDDING_MODELS option in the model select", async () => {
    render(<GraphIntelligencePage />);

    const select = await screen.findByLabelText("Model");
    for (const model of api.GRAPH_EMBEDDING_MODELS) {
      expect(within(select).getByRole("option", { name: model })).toBeInTheDocument();
    }
  });

  it("renders a button for every GRAPH_BENCHMARKS entry", async () => {
    render(<GraphIntelligencePage />);

    for (const name of api.GRAPH_BENCHMARKS) {
      expect(await screen.findByRole("button", { name })).toBeInTheDocument();
    }
  });
});

describe("GraphIntelligencePage — node feature lookup", () => {
  it("looks up features for a typed equipment name", async () => {
    const user = userEvent.setup();
    const features: NodeFeatures = {
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
    getNodeFeatures.mockResolvedValue(features);
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[0], "P-101");
    await user.click(screen.getByRole("button", { name: "Look up features" }));

    await waitFor(() => expect(getNodeFeatures).toHaveBeenCalledWith("P-101"));
    expect(await screen.findByText("0.1500")).toBeInTheDocument();
  });

  it("shows an error message when feature lookup fails", async () => {
    const user = userEvent.setup();
    getNodeFeatures.mockRejectedValue(new Error("Request failed (404): not found"));
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[0], "missing");
    await user.click(screen.getByRole("button", { name: "Look up features" }));

    expect(await screen.findByText(/Request failed \(404\)/)).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — embedding training", () => {
  it("trains embeddings with the selected model and dimensions", async () => {
    const user = userEvent.setup();
    const result: EmbeddingTrainResult = {
      model_id: "m1",
      model_name: "node2vec",
      dimensions: 16,
      node_count: 6,
      training_seconds: 0.2,
    };
    trainGraphEmbeddings.mockResolvedValue(result);
    render(<GraphIntelligencePage />);

    await user.click(await screen.findByRole("button", { name: "Train" }));

    await waitFor(() => expect(trainGraphEmbeddings).toHaveBeenCalledWith("node2vec", 32));
    expect(await screen.findByText(/Trained node2vec/)).toBeInTheDocument();
  });

  it("shows an error message when embedding training fails", async () => {
    const user = userEvent.setup();
    trainGraphEmbeddings.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<GraphIntelligencePage />);

    await user.click(await screen.findByRole("button", { name: "Train" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — similarity search", () => {
  it("finds similar nodes and shows the explanation text", async () => {
    const user = userEvent.setup();
    const neighbors: SimilarNode[] = [{ node_id: "n2", canonical_name: "V-102", score: 0.9 }];
    const explanation: GraphExplanation = {
      node_id: "n1",
      nearest_neighbors: neighbors,
      important_edges: [],
      community_id: 0,
      supporting_evidence: [],
      confidence: 0.8,
      text: "Nearest embedding neighbors: V-102 (0.9). Confidence: 80.0%.",
    };
    getSimilarNodes.mockResolvedValue(neighbors);
    explainNode.mockResolvedValue(explanation);
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[1], "P-101");
    await user.click(screen.getByRole("button", { name: "Find similar nodes" }));

    await waitFor(() => expect(getSimilarNodes).toHaveBeenCalledWith("P-101"));
    expect(await screen.findByText(/Nearest embedding neighbors/)).toBeInTheDocument();
  });

  it("shows an error message when similarity search fails (no embedding trained)", async () => {
    const user = userEvent.setup();
    getSimilarNodes.mockRejectedValue(new Error("Request failed (400): no embedding"));
    explainNode.mockRejectedValue(new Error("Request failed (400): no embedding"));
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[1], "P-101");
    await user.click(screen.getByRole("button", { name: "Find similar nodes" }));

    expect(await screen.findByText(/Request failed \(400\)/)).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — link prediction", () => {
  it("predicts link candidates for a typed equipment name", async () => {
    const user = userEvent.setup();
    const candidates: LinkPredictionCandidate[] = [
      {
        source_id: "n1",
        target_id: "n3",
        target_canonical_name: "M-103",
        confidence: 0.6,
        embedding_similarity: 0.7,
        explanation: "likely connection",
      },
    ];
    getLinkPredictionCandidates.mockResolvedValue(candidates);
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[2], "P-101");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(getLinkPredictionCandidates).toHaveBeenCalledWith("P-101"));
    expect(await screen.findByText("M-103")).toBeInTheDocument();
    expect(screen.getByText("likely connection")).toBeInTheDocument();
  });

  it("shows a message when no candidates are found", async () => {
    const user = userEvent.setup();
    getLinkPredictionCandidates.mockResolvedValue([]);
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[2], "P-101");
    await user.click(screen.getByRole("button", { name: "Predict" }));

    expect(await screen.findByText("No candidates found.")).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — risk propagation", () => {
  it("simulates propagation and renders the ranked steps", async () => {
    const user = userEvent.setup();
    const result: PropagationResult = {
      source_node_id: "n1",
      propagation_type: "failure",
      steps: [{ node_id: "n2", canonical_name: "V-102", depth: 1, propagated_risk: 0.5, path: ["n1", "n2"] }],
    };
    simulatePropagation.mockResolvedValue(result);
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[3], "P-101");
    await user.click(screen.getByRole("button", { name: "Simulate" }));

    await waitFor(() => expect(simulatePropagation).toHaveBeenCalledWith("P-101", "failure", 3));
    expect(await screen.findByText(/V-102 \(depth 1\)/)).toBeInTheDocument();
  });

  it("shows an error message when propagation fails", async () => {
    const user = userEvent.setup();
    simulatePropagation.mockRejectedValue(new Error("Request failed (404): not found"));
    render(<GraphIntelligencePage />);

    const inputs = await screen.findAllByPlaceholderText("Equipment name");
    await user.type(inputs[3], "missing");
    await user.click(screen.getByRole("button", { name: "Simulate" }));

    expect(await screen.findByText(/Request failed \(404\)/)).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — node classification", () => {
  it("trains a classifier for comma-separated equipment names and shows the registered model", async () => {
    const user = userEvent.setup();
    const result: NodeClassificationResult = {
      registered: true,
      task_type: "failure_risk",
      algorithm: "graph_RandomForestRegressor",
      sample_size: 3,
      feature_importance: [{ feature_name: "dim_0", importance: 0.4 }],
    };
    const models: ModelRecord[] = [
      {
        model_id: "m1",
        task_type: "failure_risk",
        algorithm: "graph_RandomForestRegressor",
        training_data_hash: "hash",
        hyperparameters: {},
        metrics: { r2: 0.5 },
        feature_schema: ["dim_0"],
        timestamp: "2026-01-01T00:00:00Z",
        author: "api",
        experiment_id: null,
        serialization_path: "path",
        serialization_format: "joblib",
      },
    ];
    trainNodeClassifier.mockResolvedValue(result);
    listModels.mockResolvedValue(models);
    render(<GraphIntelligencePage />);

    await user.type(screen.getByPlaceholderText("Equipment names, comma-separated"), "P-101, V-102, M-103");
    await user.click(screen.getByRole("button", { name: "Train vs. rule baseline" }));

    await waitFor(() => expect(trainNodeClassifier).toHaveBeenCalledWith("failure_risk", ["P-101", "V-102", "M-103"]));
    expect(await screen.findByText(/Registered graph_RandomForestRegressor/)).toBeInTheDocument();
    expect(await screen.findAllByText(/graph_RandomForestRegressor/)).toHaveLength(2);
  });

  it("shows a message when there isn't enough resolvable equipment to train", async () => {
    const user = userEvent.setup();
    trainNodeClassifier.mockResolvedValue({
      registered: false,
      task_type: "failure_risk",
      algorithm: null,
      sample_size: 0,
      feature_importance: [],
    });
    render(<GraphIntelligencePage />);

    await user.type(screen.getByPlaceholderText("Equipment names, comma-separated"), "only-one");
    await user.click(screen.getByRole("button", { name: "Train vs. rule baseline" }));

    expect(await screen.findByText(/Not enough resolvable equipment/)).toBeInTheDocument();
  });

  it("shows an error message when classifier training fails", async () => {
    const user = userEvent.setup();
    trainNodeClassifier.mockRejectedValue(new Error("Request failed (400): no embedding"));
    render(<GraphIntelligencePage />);

    await user.type(screen.getByPlaceholderText("Equipment names, comma-separated"), "P-101, V-102");
    await user.click(screen.getByRole("button", { name: "Train vs. rule baseline" }));

    expect(await screen.findByText(/Request failed \(400\)/)).toBeInTheDocument();
  });
});

describe("GraphIntelligencePage — research benchmarks", () => {
  it("runs a benchmark and displays its metrics and duration", async () => {
    const user = userEvent.setup();
    runBenchmark.mockResolvedValue({
      benchmark_name: "graph_embedding_node2vec",
      metrics: [{ name: "embedding_quality_margin", value: 0.05 }],
      duration_seconds: 0.3,
      dataset_name: "synthetic_ring_graph",
      sample_size: 20,
      timestamp: "2026-01-01T00:00:00Z",
    });
    render(<GraphIntelligencePage />);

    await user.click(await screen.findByRole("button", { name: "graph_embedding_node2vec" }));

    await waitFor(() => expect(runBenchmark).toHaveBeenCalledWith("graph_embedding_node2vec"));
    expect(await screen.findByText(/embedding_quality_margin=0.05/)).toBeInTheDocument();
    expect(screen.getByText("0.300s")).toBeInTheDocument();
  });

  it("shows an error message when a benchmark run fails", async () => {
    const user = userEvent.setup();
    runBenchmark.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<GraphIntelligencePage />);

    await user.click(await screen.findByRole("button", { name: "graph_link_prediction" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });
});
