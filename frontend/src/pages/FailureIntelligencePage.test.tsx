import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FailureIntelligencePage } from "./FailureIntelligencePage";
import * as api from "../services/api";
import type {
  BenchmarkReport,
  CounterfactualResult,
  ExecutionStatistics,
  FailureAttribution,
  FailurePattern,
  OptimizationRecommendation,
  RootCauseGraph,
} from "../services/api";

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
    getExecutionStatistics: vi.fn(),
    listPatterns: vi.fn(),
    getOptimizationRecommendations: vi.fn(),
    runBenchmark: vi.fn(),
    minePatterns: vi.fn(),
    getPatternAttribution: vi.fn(),
    getRootCauseGraph: vi.fn(),
    simulateCounterfactual: vi.fn(),
    getExecution: vi.fn(),
  };
});

const getExecutionStatistics = vi.mocked(api.getExecutionStatistics);
const listPatterns = vi.mocked(api.listPatterns);
const getOptimizationRecommendations = vi.mocked(api.getOptimizationRecommendations);
const runBenchmark = vi.mocked(api.runBenchmark);
const minePatterns = vi.mocked(api.minePatterns);
const getPatternAttribution = vi.mocked(api.getPatternAttribution);
const getRootCauseGraph = vi.mocked(api.getRootCauseGraph);
const simulateCounterfactual = vi.mocked(api.simulateCounterfactual);
const getExecution = vi.mocked(api.getExecution);

const STATISTICS: ExecutionStatistics = { trace_count: 10, event_count: 40, average_confidence: 0.75, average_latency_by_component: { planning: 0.05 } };
const PATTERN: FailurePattern = {
  pattern_id: "p1",
  category: "retrieval_failure",
  support: 3,
  frequency: 0.3,
  severity: "high",
  affected_components: ["HybridRetriever"],
  example_execution_ids: ["e1"],
  confidence: 0.7,
  description: "retrieval_failure occurred in 3 of 10 executions",
};
const RECOMMENDATION: OptimizationRecommendation = {
  subsystem: "HybridRetriever",
  explanation: "Increase recall",
  evidence: ["evidence line"],
  supporting_execution_ids: ["e1"],
  expected_benefit: "benefit",
  confidence: 0.7,
};
const OVERHEAD_REPORT: BenchmarkReport = {
  benchmark_name: "execution_trace_overhead",
  metrics: [{ name: "cache_hit_ratio", value: 0.5 }],
  duration_seconds: 0.01,
  dataset_name: "orchestrator_fakes",
  sample_size: 3,
  timestamp: "t",
};
const ATTRIBUTION: FailureAttribution = {
  pattern_id: "p1",
  primary_cause: "HybridRetriever",
  secondary_cause: null,
  contributing_components: [],
  evidence_confidence: 0.7,
  execution_confidence: 0.5,
  severity: "high",
  historical_frequency: 0.3,
};
const ROOT_CAUSE_GRAPH: RootCauseGraph = {
  nodes: [
    { id: "execution:e1", type: "execution", label: "q", metadata: {} },
    { id: "failure:e1:0", type: "failure", label: "retrieval_failure", metadata: {} },
    { id: "component:HybridRetriever", type: "component", label: "HybridRetriever", metadata: {} },
  ],
  edges: [{ source: "failure:e1:0", target: "component:HybridRetriever", type: "caused_by", confidence: 0.9, evidence: "e" }],
};
const COUNTERFACTUAL: CounterfactualResult = {
  execution_id: "e1",
  toggle: "verification",
  applicable: true,
  baseline_failure_categories: ["verification_failure"],
  simulated_failure_categories: [],
  disappeared_categories: ["verification_failure"],
  confidence: 0.6,
  expected_improvement: 1.0,
  note: "note text",
};

beforeEach(() => {
  getExecutionStatistics.mockResolvedValue(STATISTICS);
  listPatterns.mockResolvedValue([PATTERN]);
  getOptimizationRecommendations.mockResolvedValue([RECOMMENDATION]);
  runBenchmark.mockResolvedValue(OVERHEAD_REPORT);
  minePatterns.mockResolvedValue([PATTERN]);
  getPatternAttribution.mockResolvedValue(ATTRIBUTION);
  getRootCauseGraph.mockResolvedValue(ROOT_CAUSE_GRAPH);
  simulateCounterfactual.mockResolvedValue(COUNTERFACTUAL);
  getExecution.mockResolvedValue({
    execution_id: "e1",
    query: "q",
    events: [],
    latencies: {},
    memory: {},
    confidence: 0.5,
    timestamps: { started_at: "t0", completed_at: "t1" },
    planner_trace: null,
    retrieval_trace: null,
    ranking_trace: null,
    compression_trace: null,
    claim_extraction_trace: null,
    verification_trace: null,
    nli_trace: null,
    recommendation_trace: null,
    faithfulness_trace: null,
    reflection_trace: null,
    vision_trace: null,
    prediction_trace: null,
    graph_trace: null,
    cache_trace: null,
  });
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("FailureIntelligencePage", () => {
  it("renders every major section heading", async () => {
    render(<FailureIntelligencePage />);
    for (const heading of ["Dashboard Metrics", "Failure Pattern Explorer", "Root Cause Graph", "Counterfactual Explorer", "Optimization Recommendations"]) {
      expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
    }
  });

  it("shows dashboard metric cards computed from real data", async () => {
    render(<FailureIntelligencePage />);
    expect(await screen.findByText("Execution Count")).toBeInTheDocument();
    expect(await screen.findByText("10")).toBeInTheDocument();
    expect(await screen.findByText("Most Frequent Failure")).toBeInTheDocument();
    expect((await screen.findAllByText("retrieval_failure")).length).toBeGreaterThan(0);
  });

  it("mines patterns when the Mine Patterns button is clicked", async () => {
    const user = userEvent.setup();
    render(<FailureIntelligencePage />);
    await screen.findByText("Details");

    await user.click(screen.getByText("Mine Patterns"));
    await waitFor(() => expect(minePatterns).toHaveBeenCalled());
  });

  it("shows pattern attribution and supporting executions on Details click", async () => {
    const user = userEvent.setup();
    render(<FailureIntelligencePage />);
    await screen.findByText("Details");

    await user.click(screen.getByText("Details"));

    await waitFor(() => expect(getPatternAttribution).toHaveBeenCalledWith("p1"));
    expect(await screen.findByText("Primary cause")).toBeInTheDocument();
  });

  it("loads and displays the root cause graph for an execution id", async () => {
    const user = userEvent.setup();
    render(<FailureIntelligencePage />);

    const executionIdInputs = screen.getAllByLabelText(/Execution id/i);
    await user.type(executionIdInputs[0], "e1");
    await user.click(screen.getByText("Load Graph"));

    await waitFor(() => expect(getRootCauseGraph).toHaveBeenCalledWith("e1"));
    expect(await screen.findByText("Export JSON")).toBeInTheDocument();
  });

  it("runs a counterfactual simulation and displays the result", async () => {
    const user = userEvent.setup();
    render(<FailureIntelligencePage />);

    const executionIdInputs = screen.getAllByLabelText(/Execution id/i);
    await user.type(executionIdInputs[executionIdInputs.length - 1], "e1");
    await user.click(screen.getByText("Run Simulation"));

    await waitFor(() => expect(simulateCounterfactual).toHaveBeenCalledWith("e1", "planner"));
    expect(await screen.findByText("Disappeared:")).toBeInTheDocument();
    expect((await screen.findAllByText(/verification_failure/)).length).toBeGreaterThan(0);
  });

  it("shows optimization recommendations with drill-down evidence", async () => {
    const user = userEvent.setup();
    render(<FailureIntelligencePage />);

    expect(await screen.findByText("Increase recall")).toBeInTheDocument();
    await user.click(screen.getByText("Show supporting evidence"));
    expect(await screen.findByText("evidence line")).toBeInTheDocument();
  });
});
