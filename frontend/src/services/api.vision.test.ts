import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  VISION_BENCHMARKS,
  VISION_SOURCE_TYPES,
  analyzeVisionImage,
  getVisionImage,
  listVisionImages,
  multimodalQuery,
  uploadVisionImage,
  visionImageUrl,
} from "./api";
import type { VisionAnalysis, VisionDocument } from "./api";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as Response;
}

const SAMPLE_DOCUMENT: VisionDocument = {
  id: "img-1",
  filename: "gauge.png",
  source_type: "equipment_photo",
  content_type: "image/png",
  width: 200,
  height: 200,
  size_bytes: 4096,
  equipment_hint: "P-101",
};

const SAMPLE_ANALYSIS: VisionAnalysis = {
  image_id: "img-1",
  ocr_regions: [{ text: "P-101", confidence: 0.9, bounding_box: [0, 0, 10, 10], graph_node_id: "node-1" }],
  diagram_symbols: [],
  detections: [{ equipment_type: "P-101", confidence: 0.8, bounding_box: [5, 5, 50, 50], graph_node_id: "node-2" }],
  gauge_readings: [],
  defects: [],
};

describe("Vision API client", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("uploadVisionImage", () => {
    it("posts multipart form data with file and source_type", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_DOCUMENT));
      const file = new File(["abc"], "gauge.png", { type: "image/png" });

      const result = await uploadVisionImage(file, "equipment_photo");

      expect(result).toEqual(SAMPLE_DOCUMENT);
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toContain("/vision/images");
      expect(init?.method).toBe("POST");
      const body = init?.body as FormData;
      expect(body.get("file")).toBe(file);
      expect(body.get("source_type")).toBe("equipment_photo");
      expect(body.has("equipment_hint")).toBe(false);
    });

    it("includes equipment_hint when provided", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_DOCUMENT));
      const file = new File(["abc"], "gauge.png", { type: "image/png" });

      await uploadVisionImage(file, "equipment_photo", "P-101");

      const [, init] = vi.mocked(fetch).mock.calls[0];
      const body = init?.body as FormData;
      expect(body.get("equipment_hint")).toBe("P-101");
    });

    it("throws with status and body text on a non-ok response", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "bad source_type" }, false, 422));
      const file = new File(["abc"], "gauge.png", { type: "image/png" });

      await expect(uploadVisionImage(file, "not_a_type")).rejects.toThrow(/422/);
    });
  });

  describe("listVisionImages", () => {
    it("returns the parsed list", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse([SAMPLE_DOCUMENT]));

      const result = await listVisionImages();

      expect(result).toEqual([SAMPLE_DOCUMENT]);
    });

    it("propagates errors from a failed request", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "server error" }, false, 500));

      await expect(listVisionImages()).rejects.toThrow(/500/);
    });
  });

  describe("getVisionImage", () => {
    it("fetches a single image by id", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_DOCUMENT));

      const result = await getVisionImage("img-1");

      expect(result).toEqual(SAMPLE_DOCUMENT);
      expect(vi.mocked(fetch).mock.calls[0][0]).toContain("/vision/images/img-1");
    });

    it("throws a 404 error for an unknown image", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "not found" }, false, 404));

      await expect(getVisionImage("missing")).rejects.toThrow(/404/);
    });
  });

  describe("analyzeVisionImage", () => {
    it("posts to the analyze endpoint and returns the full analysis", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse(SAMPLE_ANALYSIS));

      const result = await analyzeVisionImage("img-1");

      expect(result).toEqual(SAMPLE_ANALYSIS);
      const [url, init] = vi.mocked(fetch).mock.calls[0];
      expect(url).toContain("/vision/images/img-1/analyze");
      expect(init?.method).toBe("POST");
    });

    it("throws a 404 error for an unknown image", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ detail: "not found" }, false, 404));

      await expect(analyzeVisionImage("missing")).rejects.toThrow(/404/);
    });
  });

  describe("multimodalQuery", () => {
    it("sends the query with the default top_k", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ evidence: [] }));

      await multimodalQuery("What is the condition of P-101?");

      const [, init] = vi.mocked(fetch).mock.calls[0];
      const body = JSON.parse(init?.body as string);
      expect(body).toEqual({ query: "What is the condition of P-101?", top_k: 8 });
    });

    it("sends a custom top_k when provided", async () => {
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ evidence: [] }));

      await multimodalQuery("query", 3);

      const [, init] = vi.mocked(fetch).mock.calls[0];
      const body = JSON.parse(init?.body as string);
      expect(body.top_k).toBe(3);
    });

    it("returns the evidence list from the response", async () => {
      const evidence = [{ source: "vision:Defect", confidence: 0.7 }];
      vi.mocked(fetch).mockResolvedValue(jsonResponse({ evidence }));

      const result = await multimodalQuery("query");

      expect(result.evidence).toEqual(evidence);
    });
  });

  describe("visionImageUrl", () => {
    it("builds the file-serving URL for an image id", () => {
      expect(visionImageUrl("img-1")).toContain("/vision/images/img-1/file");
    });
  });

  describe("VISION_SOURCE_TYPES", () => {
    it("includes every backend VisionSourceType value", () => {
      expect(VISION_SOURCE_TYPES).toEqual([
        "inspection_photo",
        "equipment_photo",
        "pid_diagram",
        "electrical_schematic",
        "engineering_drawing",
        "maintenance_image",
        "handwritten_note",
      ]);
    });
  });

  describe("VISION_BENCHMARKS", () => {
    it("includes every vision benchmark name the research API exposes", () => {
      expect(VISION_BENCHMARKS).toEqual([
        "vision_ocr",
        "vision_diagram_parsing",
        "vision_equipment_recognition",
        "vision_gauge_reading",
        "vision_defect_detection",
        "vision_multimodal_retrieval",
        "vision_latency",
      ]);
    });
  });
});
