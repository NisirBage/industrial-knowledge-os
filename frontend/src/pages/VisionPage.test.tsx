import { render, screen, waitFor, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VisionPage, formatBytes } from "./VisionPage";
import * as api from "../services/api";
import type { BenchmarkReport, Evidence, VisionAnalysis, VisionDocument } from "../services/api";

vi.mock("../services/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/api")>();
  return {
    ...actual,
    listVisionImages: vi.fn(),
    uploadVisionImage: vi.fn(),
    analyzeVisionImage: vi.fn(),
    multimodalQuery: vi.fn(),
    runBenchmark: vi.fn(),
  };
});

const listVisionImages = vi.mocked(api.listVisionImages);
const uploadVisionImage = vi.mocked(api.uploadVisionImage);
const analyzeVisionImage = vi.mocked(api.analyzeVisionImage);
const multimodalQuery = vi.mocked(api.multimodalQuery);
const runBenchmark = vi.mocked(api.runBenchmark);

const IMAGE_A: VisionDocument = {
  id: "img-a",
  filename: "gauge_a.png",
  source_type: "equipment_photo",
  content_type: "image/png",
  width: 200,
  height: 200,
  size_bytes: 2048,
  equipment_hint: "P-101",
};

const IMAGE_B: VisionDocument = {
  id: "img-b",
  filename: "diagram_b.png",
  source_type: "pid_diagram",
  content_type: "image/png",
  width: 300,
  height: 150,
  size_bytes: 5000,
  equipment_hint: null,
};

const FULL_ANALYSIS: VisionAnalysis = {
  image_id: "img-a",
  ocr_regions: [{ text: "P-101", confidence: 0.9, bounding_box: [0, 0, 20, 10], graph_node_id: "node-ocr-1" }],
  diagram_symbols: [{ symbol_type: "valve", confidence: 0.6, bounding_box: [10, 10, 40, 40], graph_node_id: "node-diagram-1" }],
  detections: [{ equipment_type: "P-101", confidence: 0.8, bounding_box: [5, 5, 50, 50], graph_node_id: "node-detection-1" }],
  gauge_readings: [
    {
      needle_angle_degrees: 0,
      normalized_value: 0.5,
      confidence: 0.6,
      valid: true,
      validation_note: "OK",
      bounding_box: [60, 60, 120, 120],
      graph_node_id: "node-gauge-1",
    },
  ],
  defects: [{ defect_type: "corrosion", confidence: 0.7, severity: "medium", bounding_box: [70, 70, 90, 90], graph_node_id: "node-defect-1" }],
};

function emptyAnalysis(imageId: string): VisionAnalysis {
  return { image_id: imageId, ocr_regions: [], diagram_symbols: [], detections: [], gauge_readings: [], defects: [] };
}

beforeEach(() => {
  listVisionImages.mockResolvedValue([]);
  uploadVisionImage.mockReset();
  analyzeVisionImage.mockReset();
  multimodalQuery.mockReset();
  runBenchmark.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("VisionPage — initial rendering", () => {
  it("renders every major section heading", async () => {
    render(<VisionPage />);

    for (const heading of ["Upload Image", "Interactive Viewer", "Analysis Results", "Knowledge Graph", "Multimodal Query", "Vision Research Benchmarks"]) {
      expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
    }
  });

  it("fetches the image list on mount", async () => {
    render(<VisionPage />);

    await waitFor(() => expect(listVisionImages).toHaveBeenCalledTimes(1));
  });

  it("shows a placeholder when no images have been uploaded", async () => {
    render(<VisionPage />);

    expect(await screen.findByText("No images uploaded yet.")).toBeInTheDocument();
  });

  it("shows an error banner when the initial image list fetch fails", async () => {
    listVisionImages.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<VisionPage />);

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });

  it("shows the viewer placeholder when no image is selected", async () => {
    render(<VisionPage />);

    expect(await screen.findByText("Upload or select an image to begin.")).toBeInTheDocument();
  });

  it("shows the analysis results placeholder when no analysis has run", async () => {
    render(<VisionPage />);

    expect(await screen.findByText(/Run analysis to see OCR/)).toBeInTheDocument();
  });

  it("shows the knowledge graph placeholder when no analysis has run", async () => {
    render(<VisionPage />);

    expect(await screen.findByText(/Run analysis to see the graph entities/)).toBeInTheDocument();
  });

  it("renders every VISION_SOURCE_TYPES option in the source type select", async () => {
    render(<VisionPage />);

    const select = await screen.findByLabelText("Source type");
    for (const sourceType of api.VISION_SOURCE_TYPES) {
      expect(within(select).getByRole("option", { name: sourceType })).toBeInTheDocument();
    }
  });

  it("renders a button for every VISION_BENCHMARKS entry", async () => {
    render(<VisionPage />);

    for (const name of api.VISION_BENCHMARKS) {
      expect(await screen.findByRole("button", { name })).toBeInTheDocument();
    }
  });

  it("renders the uploaded images list once fetched", async () => {
    listVisionImages.mockResolvedValue([IMAGE_A, IMAGE_B]);
    render(<VisionPage />);

    expect(await screen.findByText("gauge_a.png")).toBeInTheDocument();
    expect(await screen.findByText("diagram_b.png")).toBeInTheDocument();
  });

  it("shows filename, dimensions, size, source type, and analysis status for each image", async () => {
    listVisionImages.mockResolvedValue([IMAGE_A]);
    render(<VisionPage />);

    const listItem = (await screen.findByText("gauge_a.png")).closest("button") as HTMLElement;
    expect(listItem.textContent).toContain("200");
    expect(listItem.textContent).toContain("2.0 KB");
    expect(within(listItem).getByText("equipment_photo")).toBeInTheDocument();
    expect(within(listItem).getByText("Not analyzed")).toBeInTheDocument();
  });
});

describe("VisionPage — upload flow", () => {
  it("uploads a selected file with the chosen source type", async () => {
    const user = userEvent.setup();
    uploadVisionImage.mockResolvedValue(IMAGE_A);
    listVisionImages.mockResolvedValue([]);
    render(<VisionPage />);

    const file = new File(["abc"], "test.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    await waitFor(() => expect(uploadVisionImage).toHaveBeenCalledWith(file, "inspection_photo", undefined));
  });

  it("includes the equipment hint when the field is filled in", async () => {
    const user = userEvent.setup();
    uploadVisionImage.mockResolvedValue(IMAGE_A);
    render(<VisionPage />);

    await user.type(await screen.findByLabelText("Equipment hint (optional)"), "P-999");
    const file = new File(["abc"], "test.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    await waitFor(() => expect(uploadVisionImage).toHaveBeenCalledWith(file, "inspection_photo", "P-999"));
  });

  it("uses the selected source type from the dropdown", async () => {
    const user = userEvent.setup();
    uploadVisionImage.mockResolvedValue(IMAGE_A);
    render(<VisionPage />);

    await user.selectOptions(await screen.findByLabelText("Source type"), "pid_diagram");
    const file = new File(["abc"], "test.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    await waitFor(() => expect(uploadVisionImage).toHaveBeenCalledWith(file, "pid_diagram", undefined));
  });

  it("rejects an unsupported file extension without calling the API", async () => {
    render(<VisionPage />);

    const file = new File(["abc"], "malware.exe", { type: "application/octet-stream" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });

    expect(await screen.findByText(/Unsupported file type/)).toBeInTheDocument();
    expect(uploadVisionImage).not.toHaveBeenCalled();
  });

  it("shows an error message when the upload fails", async () => {
    const user = userEvent.setup();
    uploadVisionImage.mockRejectedValue(new Error("Request failed (422): bad source_type"));
    render(<VisionPage />);

    const file = new File(["abc"], "test.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    expect(await screen.findByText(/Request failed \(422\)/)).toBeInTheDocument();
  });

  it("refreshes the image list and selects the new image after a successful upload", async () => {
    uploadVisionImage.mockResolvedValue(IMAGE_A);
    listVisionImages.mockResolvedValueOnce([]).mockResolvedValueOnce([IMAGE_A]);
    render(<VisionPage />);

    const file = new File(["abc"], "test.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => expect(listVisionImages).toHaveBeenCalledTimes(2));
    expect(await screen.findByAltText("gauge_a.png")).toBeInTheDocument();
  });

  it("uploads a file dropped onto the dropzone", async () => {
    uploadVisionImage.mockResolvedValue(IMAGE_A);
    render(<VisionPage />);

    const dropzone = await screen.findByTestId("vision-dropzone");
    const file = new File(["abc"], "dropped.png", { type: "image/png" });
    const dataTransfer = { files: [file] };

    fireEvent.drop(dropzone, { dataTransfer });

    await waitFor(() => expect(uploadVisionImage).toHaveBeenCalledWith(file, "inspection_photo", undefined));
  });
});

describe("VisionPage — image selection", () => {
  it("selects an image and renders it in the viewer", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));

    expect(await screen.findByAltText("gauge_a.png")).toBeInTheDocument();
    expect(screen.queryByText("Upload or select an image to begin.")).not.toBeInTheDocument();
  });

  it("resets analysis state when switching to a different image", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A, IMAGE_B]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));
    await screen.findByText(/OCR complete/);

    await user.click(screen.getByText("diagram_b.png"));

    expect(screen.queryByText(/OCR complete/)).not.toBeInTheDocument();
    expect(screen.getByText(/Run analysis to see OCR/)).toBeInTheDocument();
  });
});

describe("VisionPage — analysis pipeline", () => {
  it("disables Run Analysis until an image is selected", async () => {
    render(<VisionPage />);

    expect(await screen.findByRole("button", { name: "Run Analysis" })).toBeDisabled();
  });

  it("calls analyzeVisionImage with the selected image's id", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(emptyAnalysis("img-a"));
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    await waitFor(() => expect(analyzeVisionImage).toHaveBeenCalledWith("img-a"));
  });

  it("shows a completion checklist with correct counts after analysis", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    const checklist = await screen.findByText(/OCR complete/);
    expect(checklist.textContent).toContain("1 region(s)");
    expect(checklist.textContent).toContain("1 object(s)");
    expect(checklist.textContent).toContain("1 symbol(s)");
    expect(checklist.textContent).toContain("1 gauge(s)");
    expect(checklist.textContent).toContain("1 defect(s)");
    expect(checklist.textContent).toContain("Graph integration complete");
  });

  it("marks the image as Analyzed in the upload list after a successful run", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(emptyAnalysis("img-a"));
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(await screen.findByText("Analyzed")).toBeInTheDocument();
  });

  it("shows an error message when analysis fails", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockRejectedValue(new Error("Request failed (404): not found"));
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(await screen.findByText(/Request failed \(404\)/)).toBeInTheDocument();
  });

  it("renders each non-empty result category with its item count", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(await screen.findByText("OCR Regions (1)")).toBeInTheDocument();
    expect(screen.getByText("Detections (1)")).toBeInTheDocument();
    expect(screen.getByText("Diagram Symbols (1)")).toBeInTheDocument();
    expect(screen.getByText("Gauge Readings (1)")).toBeInTheDocument();
    expect(screen.getByText("Defects (1)")).toBeInTheDocument();
  });

  it("omits empty result categories entirely", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(emptyAnalysis("img-a"));
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    await screen.findByText(/OCR complete/);
    expect(screen.queryByText(/OCR Regions/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Detections \(/)).not.toBeInTheDocument();
  });

  it("shows the gauge's uncalibrated normalized value and validation note", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(await screen.findByText(/50% of scale \(uncalibrated\)/)).toBeInTheDocument();
    expect(screen.getByText(/validated: yes \(OK\)/)).toBeInTheDocument();
  });
});

describe("VisionPage — interactive viewer overlays and selection", () => {
  async function renderAnalyzed() {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);
    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));
    await screen.findByText(/OCR complete/);
    return user;
  }

  it("renders one overlay element per detected item", async () => {
    await renderAnalyzed();

    expect(document.querySelector('[data-testid="ocr-0"]')).toBeInTheDocument();
    expect(document.querySelector('[data-testid="detection-0"]')).toBeInTheDocument();
    expect(document.querySelector('[data-testid="diagram-0"]')).toBeInTheDocument();
    expect(document.querySelector('[data-testid="gauge-0"]')).toBeInTheDocument();
    expect(document.querySelector('[data-testid="defect-0"]')).toBeInTheDocument();
  });

  it("hides a layer's overlays when its checkbox is unchecked", async () => {
    const user = await renderAnalyzed();

    await user.click(screen.getByRole("checkbox", { name: /OCR/ }));

    expect(document.querySelector('[data-testid="ocr-0"]')).not.toBeInTheDocument();
    expect(document.querySelector('[data-testid="detection-0"]')).toBeInTheDocument();
  });

  it("increases stroke-width when an overlay box is clicked (selected)", async () => {
    await renderAnalyzed();
    const rect = document.querySelector('[data-testid="detection-0"]') as SVGRectElement;
    expect(rect.getAttribute("stroke-width")).toBe("2");

    fireEvent.click(rect);

    expect(rect.getAttribute("stroke-width")).toBe("4");
  });

  it("toggles selection off when the same overlay box is clicked twice", async () => {
    await renderAnalyzed();
    const rect = document.querySelector('[data-testid="detection-0"]') as SVGRectElement;

    fireEvent.click(rect);
    fireEvent.click(rect);

    expect(rect.getAttribute("stroke-width")).toBe("2");
  });

  it("clicking a result-list item highlights the same key as the overlay", async () => {
    const user = await renderAnalyzed();
    const detectionButton = screen.getByRole("button", { name: /P-101 80%/ });

    await user.click(detectionButton);

    expect(detectionButton.className).toContain("border-indigo-400");
    const rect = document.querySelector('[data-testid="detection-0"]') as SVGRectElement;
    expect(rect.getAttribute("stroke-width")).toBe("4");
  });

  it("zoom in increases the displayed percentage", async () => {
    const user = await renderAnalyzed();

    await user.click(screen.getByRole("button", { name: "Zoom in" }));

    expect(screen.getByTestId("zoom-level")).toHaveTextContent("125%");
  });

  it("zoom out decreases the displayed percentage, floored at 50%", async () => {
    const user = await renderAnalyzed();

    for (let i = 0; i < 4; i++) {
      await user.click(screen.getByRole("button", { name: "Zoom out" }));
    }

    expect(screen.getByTestId("zoom-level")).toHaveTextContent("50%");
  });

  it("Fit to screen resets zoom back to 100%", async () => {
    const user = await renderAnalyzed();
    await user.click(screen.getByRole("button", { name: "Zoom in" }));
    await user.click(screen.getByRole("button", { name: "Zoom in" }));

    await user.click(screen.getByRole("button", { name: "Fit to screen" }));

    expect(screen.getByTestId("zoom-level")).toHaveTextContent("100%");
  });
});

describe("VisionPage — Knowledge Graph panel", () => {
  it("renders one row per analysis item with its graph node id", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));

    expect(await screen.findByText("OCR Region")).toBeInTheDocument();
    expect(screen.getByText("Detected Object")).toBeInTheDocument();
    expect(screen.getByText("Diagram Symbol")).toBeInTheDocument();
    expect(screen.getByText("Gauge Reading")).toBeInTheDocument();
    expect(screen.getByText("Defect")).toBeInTheDocument();
    expect(screen.getByText("node-ocr-1".slice(0, 8))).toBeInTheDocument();
  });

  it("selecting a knowledge graph row highlights the corresponding overlay", async () => {
    const user = userEvent.setup();
    listVisionImages.mockResolvedValue([IMAGE_A]);
    analyzeVisionImage.mockResolvedValue(FULL_ANALYSIS);
    render(<VisionPage />);

    await user.click(await screen.findByText("gauge_a.png"));
    await user.click(screen.getByRole("button", { name: "Run Analysis" }));
    const row = (await screen.findByText("Detected Object")).closest("tr") as HTMLElement;

    await user.click(row);

    const rect = document.querySelector('[data-testid="detection-0"]') as SVGRectElement;
    expect(rect.getAttribute("stroke-width")).toBe("4");
  });
});

describe("VisionPage — multimodal query", () => {
  it("disables Ask until a query is typed", async () => {
    render(<VisionPage />);

    expect(await screen.findByRole("button", { name: "Ask" })).toBeDisabled();
  });

  it("calls multimodalQuery with the typed text when Ask is clicked", async () => {
    const user = userEvent.setup();
    multimodalQuery.mockResolvedValue({ evidence: [] });
    render(<VisionPage />);

    await user.type(screen.getByPlaceholderText(/gauge reading/), "What defects were detected?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    await waitFor(() => expect(multimodalQuery).toHaveBeenCalledWith("What defects were detected?"));
  });

  it("submits the query when Enter is pressed in the input", async () => {
    const user = userEvent.setup();
    multimodalQuery.mockResolvedValue({ evidence: [] });
    render(<VisionPage />);

    const input = screen.getByPlaceholderText(/gauge reading/);
    await user.type(input, "What is the gauge reading?{Enter}");

    await waitFor(() => expect(multimodalQuery).toHaveBeenCalledWith("What is the gauge reading?"));
  });

  it("shows a message when no evidence is found", async () => {
    const user = userEvent.setup();
    multimodalQuery.mockResolvedValue({ evidence: [] });
    render(<VisionPage />);

    await user.type(screen.getByPlaceholderText(/gauge reading/), "anything");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("No evidence found for this query.")).toBeInTheDocument();
  });

  it("labels vision-sourced evidence as Visual with its node type", async () => {
    const user = userEvent.setup();
    const evidence: Evidence[] = [
      {
        chunk_id: "c1",
        document_id: "d1",
        text: "A defect was found.",
        page_number: 0,
        entity_ids: [],
        relation_ids: [],
        graph_score: 0,
        vector_score: 0,
        bm25_score: 0,
        reranker_score: 0,
        source: "vision:Defect",
        confidence: 0.7,
        agent: "",
      },
    ];
    multimodalQuery.mockResolvedValue({ evidence });
    render(<VisionPage />);

    await user.type(screen.getByPlaceholderText(/gauge reading/), "defects?");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Visual — Defect")).toBeInTheDocument();
    expect(screen.getByText("A defect was found.")).toBeInTheDocument();
  });

  it("labels fusion-sourced evidence as Text", async () => {
    const user = userEvent.setup();
    const evidence: Evidence[] = [
      {
        chunk_id: "c1",
        document_id: "d1",
        text: "Some document text.",
        page_number: 0,
        entity_ids: [],
        relation_ids: [],
        graph_score: 0,
        vector_score: 0,
        bm25_score: 0,
        reranker_score: 0,
        source: "fusion",
        confidence: 0.5,
        agent: "",
      },
    ];
    multimodalQuery.mockResolvedValue({ evidence });
    render(<VisionPage />);

    await user.type(screen.getByPlaceholderText(/gauge reading/), "anything");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("Text")).toBeInTheDocument();
  });

  it("shows an error message when the query fails", async () => {
    const user = userEvent.setup();
    multimodalQuery.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<VisionPage />);

    await user.type(screen.getByPlaceholderText(/gauge reading/), "anything");
    await user.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });
});

describe("VisionPage — research benchmarks", () => {
  it("runs a benchmark and displays its metrics and duration", async () => {
    const user = userEvent.setup();
    const report: BenchmarkReport = {
      benchmark_name: "vision_gauge_reading",
      metrics: [{ name: "mean_absolute_error", value: 0.01 }],
      duration_seconds: 0.112,
      dataset_name: "synthetic_gauges",
      sample_size: 5,
      timestamp: "2026-01-01T00:00:00Z",
    };
    runBenchmark.mockResolvedValue(report);
    render(<VisionPage />);

    await user.click(await screen.findByRole("button", { name: "vision_gauge_reading" }));

    await waitFor(() => expect(runBenchmark).toHaveBeenCalledWith("vision_gauge_reading"));
    expect(await screen.findByText(/mean_absolute_error=0.01/)).toBeInTheDocument();
    expect(screen.getByText("0.112s")).toBeInTheDocument();
  });

  it("shows an error message when a benchmark run fails", async () => {
    const user = userEvent.setup();
    runBenchmark.mockRejectedValue(new Error("Request failed (500): boom"));
    render(<VisionPage />);

    await user.click(await screen.findByRole("button", { name: "vision_ocr" }));

    expect(await screen.findByText(/Request failed \(500\)/)).toBeInTheDocument();
  });

  it("only disables the benchmark button that is currently running", async () => {
    const user = userEvent.setup();
    let resolveRun: (value: BenchmarkReport) => void = () => undefined;
    runBenchmark.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveRun = resolve;
        }),
    );
    render(<VisionPage />);

    await user.click(await screen.findByRole("button", { name: "vision_ocr" }));

    expect(screen.getByRole("button", { name: "Running..." })).toBeDisabled();
    expect(screen.getByRole("button", { name: "vision_diagram_parsing" })).not.toBeDisabled();

    resolveRun({
      benchmark_name: "vision_ocr",
      metrics: [],
      duration_seconds: 0.01,
      dataset_name: "synthetic",
      sample_size: 1,
      timestamp: "2026-01-01T00:00:00Z",
    });
  });
});

describe("formatBytes", () => {
  it("formats sub-kilobyte sizes in bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
  });

  it("formats kilobyte-scale sizes", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
  });

  it("formats megabyte-scale sizes", () => {
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});
