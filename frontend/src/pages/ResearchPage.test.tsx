import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ResearchPage } from "./ResearchPage";
import * as api from "../services/api";

vi.mock("../services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/api")>();
  return {
    ...actual,
    listExperiments: vi.fn(),
    getBenchmarkHistory: vi.fn(),
    getCalibration: vi.fn(),
    runBenchmark: vi.fn(),
    runEntityExtractionABTest: vi.fn(),
    previewContextCompression: vi.fn(),
    getEntityDuplicates: vi.fn(),
    mergeEntities: vi.fn(),
  };
});

const listExperiments = vi.mocked(api.listExperiments);
const getBenchmarkHistory = vi.mocked(api.getBenchmarkHistory);
const getCalibration = vi.mocked(api.getCalibration);
const runBenchmark = vi.mocked(api.runBenchmark);
const previewContextCompression = vi.mocked(api.previewContextCompression);
const getEntityDuplicates = vi.mocked(api.getEntityDuplicates);
const mergeEntities = vi.mocked(api.mergeEntities);

beforeEach(() => {
  listExperiments.mockResolvedValue([]);
  getBenchmarkHistory.mockResolvedValue([]);
  getCalibration.mockResolvedValue({ buckets: [] });
  runBenchmark.mockReset();
  previewContextCompression.mockReset();
  getEntityDuplicates.mockReset();
  mergeEntities.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("ResearchPage — Optimization Benchmarks section", () => {
  it("renders a button for every OPTIMIZATION_BENCHMARKS entry", async () => {
    render(<ResearchPage />);

    for (const name of api.OPTIMIZATION_BENCHMARKS) {
      expect(await screen.findByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("runs an optimization benchmark and displays its metrics and duration", async () => {
    const user = userEvent.setup();
    runBenchmark.mockResolvedValue({
      benchmark_name: "context_compression",
      metrics: [{ name: "compression_ratio", value: 0.71 }],
      duration_seconds: 0.05,
      dataset_name: "synthetic_evidence_with_duplicates",
      sample_size: 14,
      timestamp: "2026-01-01T00:00:00Z",
    });
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "context_compression" }));

    expect(await screen.findByText(/compression_ratio=0.71/)).toBeInTheDocument();
    expect(screen.getByText("0.050s")).toBeInTheDocument();
  });

  it("shows an error message when an optimization benchmark run fails", async () => {
    const user = userEvent.setup();
    runBenchmark.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "ranking_quality" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });

  it("only disables the optimization benchmark button that is currently running", async () => {
    const user = userEvent.setup();
    let resolveRun: (value: Awaited<ReturnType<typeof api.runBenchmark>>) => void = () => undefined;
    runBenchmark.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRun = resolve;
        }),
    );
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "ranking_quality" }));

    expect(screen.getByRole("button", { name: "Running..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "context_compression" })).not.toBeDisabled();

    resolveRun({
      benchmark_name: "ranking_quality",
      metrics: [],
      duration_seconds: 0.01,
      dataset_name: "synthetic",
      sample_size: 1,
      timestamp: "2026-01-01T00:00:00Z",
    });
  });
});

describe("ResearchPage — Reasoning Benchmarks section", () => {
  it("renders a button for every REASONING_BENCHMARKS entry", async () => {
    render(<ResearchPage />);

    for (const name of api.REASONING_BENCHMARKS) {
      expect(await screen.findByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("runs a reasoning benchmark and displays its metrics and duration", async () => {
    const user = userEvent.setup();
    runBenchmark.mockResolvedValue({
      benchmark_name: "faithfulness_quality",
      metrics: [{ name: "correctly_ranked_faithful_above_hallucinated", value: 1.0 }],
      duration_seconds: 0.02,
      dataset_name: "synthetic_faithful_vs_hallucinated",
      sample_size: 2,
      timestamp: "2026-01-01T00:00:00Z",
    });
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "faithfulness_quality" }));

    expect(await screen.findByText(/correctly_ranked_faithful_above_hallucinated=1/)).toBeInTheDocument();
    expect(screen.getByText("0.020s")).toBeInTheDocument();
  });

  it("shows an error message when a reasoning benchmark run fails", async () => {
    const user = userEvent.setup();
    runBenchmark.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<ResearchPage />);

    await user.click(await screen.findByRole("button", { name: "nli_quality" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });
});

describe("ResearchPage — Context Compression Preview", () => {
  it("previews compression for a query and shows the summary", async () => {
    const user = userEvent.setup();
    previewContextCompression.mockResolvedValue({
      clusters: [],
      compressed_evidence: [],
      stats: {
        original_count: 14,
        compressed_count: 8,
        exact_duplicate_count: 4,
        near_duplicate_count: 2,
        cluster_count: 8,
        compression_ratio: 0.5714,
        original_tokens_estimate: 900,
        compressed_tokens_estimate: 500,
        token_reduction_ratio: 0.4444,
        summary: "Compressed 21 chunks into 8 evidence groups (8 clusters, 4 exact + 2 near duplicates removed, ~44% token reduction).",
      },
    });
    render(<ResearchPage />);

    await user.type(screen.getByPlaceholderText(/pump P-101/), "pump P-101 failures");
    await user.click(screen.getByRole("button", { name: "Preview Compression" }));

    expect(await screen.findByText(/Compressed 21 chunks into 8 evidence groups/)).toBeInTheDocument();
    expect(previewContextCompression).toHaveBeenCalledWith("pump P-101 failures", "balanced");
  });
});

describe("ResearchPage — Entity Resolution duplicate candidates", () => {
  it("finds duplicates and merges a pair", async () => {
    const user = userEvent.setup();
    getEntityDuplicates.mockResolvedValue([
      {
        source_id: "src-1",
        source_name: "P-101",
        target_id: "tgt-1",
        target_name: "PUMP P-101",
        confidence: 0.92,
        merge_reason: "edit_distance=0.90",
      },
    ]);
    mergeEntities.mockResolvedValue({
      merge_id: "merge-1",
      source_id: "src-1",
      target_id: "tgt-1",
      reason: "manual",
      timestamp: "2026-01-01T00:00:00Z",
    });
    render(<ResearchPage />);

    await user.click(screen.getByRole("button", { name: "Find Duplicates" }));
    expect(await screen.findByText("PUMP P-101")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Merge" }));

    expect(await screen.findByText("Merged")).toBeInTheDocument();
    expect(mergeEntities).toHaveBeenCalledWith("src-1", "tgt-1");
  });
});
