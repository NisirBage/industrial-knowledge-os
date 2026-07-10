import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ExecutionsPage } from "./ExecutionsPage";
import * as api from "../services/api";
import type { ExecutionTrace, FailurePattern, ReconstructedTimeline } from "../services/api";

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
    listExecutions: vi.fn(),
    listPatterns: vi.fn(),
    getExecution: vi.fn(),
    getExecutionTimeline: vi.fn(),
  };
});

const listExecutions = vi.mocked(api.listExecutions);
const listPatterns = vi.mocked(api.listPatterns);
const getExecution = vi.mocked(api.getExecution);
const getExecutionTimeline = vi.mocked(api.getExecutionTimeline);

function trace(overrides: Partial<ExecutionTrace> = {}): ExecutionTrace {
  return {
    execution_id: "e1",
    query: "Why did P-101 fail?",
    events: [
      { event_type: "planning_started", component: "planning", timestamp: "t0", duration: 0, inputs: {}, outputs: {}, metadata: {}, confidence: null, latency: 0, memory_bytes: null },
      { event_type: "planning_finished", component: "planning", timestamp: "t1", duration: 0.1, inputs: {}, outputs: {}, metadata: {}, confidence: 0.8, latency: 0.1, memory_bytes: 1024 },
    ],
    latencies: { planning: 0.1 },
    memory: { planning: 1024 },
    confidence: 0.8,
    timestamps: { started_at: "2026-01-01T00:00:00", completed_at: "2026-01-01T00:00:01" },
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
    ...overrides,
  };
}

const PATTERN: FailurePattern = {
  pattern_id: "p1",
  category: "retrieval_failure",
  support: 1,
  frequency: 0.5,
  severity: "high",
  affected_components: ["HybridRetriever"],
  example_execution_ids: ["e2"],
  confidence: 0.7,
  description: "d",
};

const TIMELINE: ReconstructedTimeline = {
  spans: [{ component: "planning", started_at: "2026-01-01T00:00:00", finished_at: "2026-01-01T00:00:01", duration_seconds: 0.1, confidence: 0.8, memory_bytes: 1024 }],
  component_order: ["planning"],
  parallel_branches: [],
  cache_hits: [],
  confidence_evolution: [["planning", 0.8]],
  dag: { nodes: ["planning"], edges: [] },
};

beforeEach(() => {
  listExecutions.mockResolvedValue([trace(), trace({ execution_id: "e2", query: "Why did V-2 leak?" })]);
  listPatterns.mockResolvedValue([PATTERN]);
  getExecution.mockResolvedValue(trace());
  getExecutionTimeline.mockResolvedValue(TIMELINE);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("ExecutionsPage", () => {
  it("fetches and lists executions on mount", async () => {
    render(<ExecutionsPage />);
    await waitFor(() => expect(listExecutions).toHaveBeenCalledWith(100));
    expect(await screen.findByText("Why did P-101 fail?")).toBeInTheDocument();
    expect(await screen.findByText("Why did V-2 leak?")).toBeInTheDocument();
  });

  it("filters executions by search text", async () => {
    const user = userEvent.setup();
    render(<ExecutionsPage />);
    await screen.findByText("Why did P-101 fail?");

    await user.type(screen.getByLabelText(/Search/), "V-2");

    expect(screen.queryByText("Why did P-101 fail?")).not.toBeInTheDocument();
    expect(screen.getByText("Why did V-2 leak?")).toBeInTheDocument();
  });

  it("opens the execution inspector when View is clicked", async () => {
    const user = userEvent.setup();
    render(<ExecutionsPage />);
    await screen.findByText("Why did P-101 fail?");

    const viewButtons = screen.getAllByText("View");
    await user.click(viewButtons[0]);

    await waitFor(() => expect(getExecution).toHaveBeenCalled());
    await waitFor(() => expect(getExecutionTimeline).toHaveBeenCalled());
    expect(await screen.findByText(/Execution Inspector/)).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Pipeline Waterfall" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Execution DAG" })).toBeInTheDocument();
  });

  it("looks up an execution directly by id", async () => {
    const user = userEvent.setup();
    render(<ExecutionsPage />);
    await screen.findByText("Why did P-101 fail?");

    await user.type(screen.getByLabelText(/Look up execution id/), "e1");
    await user.click(screen.getByText("Open"));

    await waitFor(() => expect(getExecution).toHaveBeenCalledWith("e1"));
    expect(await screen.findByText(/Execution Inspector/)).toBeInTheDocument();
  });
});
