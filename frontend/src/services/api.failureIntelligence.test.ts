import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  getClusters,
  getOptimizationRecommendations,
  getPatternAttribution,
  getRootCauseGraph,
  listPatterns,
  minePatterns,
  simulateCounterfactual,
} from "./api";
import type { CounterfactualResult, FailureAttribution, FailureCluster, FailurePattern, OptimizationRecommendation, RootCauseGraph } from "./api";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response;
}

const SAMPLE_PATTERN: FailurePattern = {
  pattern_id: "p1",
  category: "retrieval_failure",
  support: 3,
  frequency: 0.5,
  severity: "high",
  affected_components: ["HybridRetriever"],
  example_execution_ids: ["e1", "e2"],
  confidence: 0.7,
  description: "d",
};

const SAMPLE_ATTRIBUTION: FailureAttribution = {
  pattern_id: "p1",
  primary_cause: "HybridRetriever",
  secondary_cause: null,
  contributing_components: [],
  evidence_confidence: 0.7,
  execution_confidence: 0.5,
  severity: "high",
  historical_frequency: 0.5,
};

const SAMPLE_CLUSTER: FailureCluster = {
  cluster_id: "rule_based:category:0",
  strategy_name: "rule_based",
  key: "retrieval_failure",
  execution_ids: ["e1", "e2"],
  size: 2,
};

const SAMPLE_GRAPH: RootCauseGraph = {
  nodes: [{ id: "execution:e1", type: "execution", label: "q", metadata: {} }],
  edges: [],
};

const SAMPLE_COUNTERFACTUAL: CounterfactualResult = {
  execution_id: "e1",
  toggle: "verification",
  applicable: true,
  baseline_failure_categories: ["verification_failure"],
  simulated_failure_categories: [],
  disappeared_categories: ["verification_failure"],
  confidence: 0.6,
  expected_improvement: 1.0,
  note: "note",
};

const SAMPLE_RECOMMENDATION: OptimizationRecommendation = {
  subsystem: "HybridRetriever",
  explanation: "explanation",
  evidence: ["evidence"],
  supporting_execution_ids: ["e1"],
  expected_benefit: "benefit",
  confidence: 0.7,
};

describe("Failure Intelligence API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("minePatterns POSTs with a limit query param", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([SAMPLE_PATTERN]));
    const result = await minePatterns(50);
    expect(result).toEqual([SAMPLE_PATTERN]);
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toContain("/failure-intelligence/patterns/mine?limit=50");
    expect(init?.method).toBe("POST");
  });

  it("listPatterns fetches all persisted patterns", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([SAMPLE_PATTERN]));
    const result = await listPatterns();
    expect(result).toEqual([SAMPLE_PATTERN]);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/failure-intelligence/patterns");
  });

  it("getPatternAttribution fetches by encoded pattern id", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_ATTRIBUTION));
    const result = await getPatternAttribution("p 1");
    expect(result).toEqual(SAMPLE_ATTRIBUTION);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/failure-intelligence/patterns/p%201/attribution");
  });

  it("getClusters passes strategy/key/limit query params", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([SAMPLE_CLUSTER]));
    const result = await getClusters("rule_based", "component_sequence", 25);
    expect(result).toEqual([SAMPLE_CLUSTER]);
    const url = vi.mocked(fetch).mock.calls[0][0] as string;
    expect(url).toContain("strategy=rule_based");
    expect(url).toContain("key=component_sequence");
    expect(url).toContain("limit=25");
  });

  it("getRootCauseGraph fetches by encoded execution id", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_GRAPH));
    const result = await getRootCauseGraph("e1");
    expect(result).toEqual(SAMPLE_GRAPH);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/failure-intelligence/root-cause-graph/e1");
  });

  it("simulateCounterfactual POSTs the toggle as a JSON body", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse(SAMPLE_COUNTERFACTUAL));
    const result = await simulateCounterfactual("e1", "verification");
    expect(result).toEqual(SAMPLE_COUNTERFACTUAL);
    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toContain("/failure-intelligence/counterfactual/e1");
    expect(JSON.parse(init?.body as string)).toEqual({ toggle: "verification" });
  });

  it("getOptimizationRecommendations fetches the recommendation list", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse([SAMPLE_RECOMMENDATION]));
    const result = await getOptimizationRecommendations();
    expect(result).toEqual([SAMPLE_RECOMMENDATION]);
    expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/failure-intelligence/optimization-recommendations");
  });

  it("throws a descriptive error when the response is not ok", async () => {
    vi.mocked(fetch).mockResolvedValueOnce(jsonResponse("boom", false, 500));
    await expect(listPatterns()).rejects.toThrow("Request failed (500)");
  });
});
