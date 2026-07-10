import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getExecution, getExecutionStatistics, getExecutionTimeline, listExecutions } from "./api";
import type { ExecutionStatistics, ExecutionTrace, ReconstructedTimeline } from "./api";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response;
}

const SAMPLE_TRACE: ExecutionTrace = {
  execution_id: "e1",
  query: "Why did P-101 fail?",
  events: [
    {
      event_type: "planning_started",
      component: "planning",
      timestamp: "2026-01-01T00:00:00+00:00",
      duration: 0,
      inputs: {},
      outputs: {},
      metadata: {},
      confidence: null,
      latency: 0,
      memory_bytes: null,
    },
  ],
  latencies: { planning: 0.05 },
  memory: { planning: 1024 },
  confidence: 0.8,
  timestamps: { started_at: "2026-01-01T00:00:00+00:00", completed_at: "2026-01-01T00:00:01+00:00" },
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
};

const SAMPLE_TIMELINE: ReconstructedTimeline = {
  spans: [{ component: "planning", started_at: "t0", finished_at: "t1", duration_seconds: 0.05, confidence: null, memory_bytes: null }],
  component_order: ["planning"],
  parallel_branches: [],
  cache_hits: [],
  confidence_evolution: [],
  dag: { nodes: ["planning"], edges: [] },
};

const SAMPLE_STATISTICS: ExecutionStatistics = {
  trace_count: 5,
  event_count: 20,
  average_confidence: 0.75,
  average_latency_by_component: { planning: 0.05 },
};

describe("Execution Trace API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("listExecutions fetches with a limit query param", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([SAMPLE_TRACE]));
    const result = await listExecutions(25);
    expect(result).toEqual([SAMPLE_TRACE]);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/executions?limit=25");
  });

  it("getExecution fetches by encoded execution id", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_TRACE));
    const result = await getExecution("e 1");
    expect(result).toEqual(SAMPLE_TRACE);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/executions/e%201");
  });

  it("getExecutionTimeline fetches the timeline sub-resource", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_TIMELINE));
    const result = await getExecutionTimeline("e1");
    expect(result).toEqual(SAMPLE_TIMELINE);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/executions/e1/timeline");
  });

  it("getExecutionStatistics fetches aggregate statistics", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_STATISTICS));
    const result = await getExecutionStatistics();
    expect(result).toEqual(SAMPLE_STATISTICS);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/executions/statistics");
  });

  it("throws a descriptive error when the response is not ok", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse("not found", false, 404));
    await expect(getExecution("missing")).rejects.toThrow("Request failed (404)");
  });
});
