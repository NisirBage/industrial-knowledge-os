import { Component, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent, ReactNode } from "react";
import {
  VISION_BENCHMARKS,
  VISION_SOURCE_TYPES,
  analyzeVisionImage,
  listVisionImages,
  multimodalQuery,
  runBenchmark,
  uploadVisionImage,
  visionImageUrl,
} from "../services/api";
import type { BenchmarkReport, Evidence, VisionAnalysis, VisionDocument } from "../services/api";

type LayerKey = "ocr" | "detection" | "diagram" | "gauge" | "defect";

const LAYER_COLORS: Record<LayerKey, string> = {
  detection: "#4f46e5",
  ocr: "#059669",
  diagram: "#7c3aed",
  gauge: "#d97706",
  defect: "#dc2626",
};

const LAYER_LABELS: Record<LayerKey, string> = {
  detection: "Equipment",
  ocr: "OCR",
  diagram: "Diagram Symbols",
  gauge: "Gauges",
  defect: "Defects",
};

const ACCEPTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"];
const MAX_VISIBLE_LIST_ITEMS = 50;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.7) return "bg-emerald-50 text-emerald-700";
  if (confidence >= 0.4) return "bg-amber-50 text-amber-700";
  return "bg-red-50 text-red-700";
}

function evidenceSourceBadge(source: string): { label: string; className: string } {
  if (source.startsWith("vision:")) {
    return { label: `Visual — ${source.slice("vision:".length)}`, className: "bg-violet-50 text-violet-700" };
  }
  if (source === "graph") {
    return { label: "Graph", className: "bg-indigo-50 text-indigo-700" };
  }
  return { label: "Text", className: "bg-slate-100 text-slate-700" };
}

interface ErrorBoundaryProps {
  children: ReactNode;
}
interface ErrorBoundaryState {
  error: Error | null;
}

class VisionErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-4xl p-6">
          <p className="rounded-lg bg-red-50 p-3 text-red-700">
            The Vision Workspace crashed: {this.state.error.message}. Try reloading the page.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

function VisionWorkspace() {
  const [images, setImages] = useState<VisionDocument[]>([]);
  const [selectedImageId, setSelectedImageId] = useState<string | null>(null);
  const [sourceType, setSourceType] = useState<string>(VISION_SOURCE_TYPES[0]);
  const [equipmentHint, setEquipmentHint] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<VisionAnalysis | null>(null);
  const [layerVisibility, setLayerVisibility] = useState<Record<LayerKey, boolean>>({
    detection: true,
    ocr: true,
    diagram: true,
    gauge: true,
    defect: true,
  });
  const [selectedItemKey, setSelectedItemKey] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const [query, setQuery] = useState("");
  const [queryResult, setQueryResult] = useState<Evidence[] | null>(null);

  const [benchmarkResults, setBenchmarkResults] = useState<Record<string, BenchmarkReport>>({});
  const [expandedLists, setExpandedLists] = useState<Record<string, boolean>>({});

  const fileInputRef = useRef<HTMLInputElement>(null);

  const selectedImage = useMemo(() => images.find((img) => img.id === selectedImageId) ?? null, [images, selectedImageId]);

  async function refreshImages() {
    setImages(await listVisionImages());
  }

  useEffect(() => {
    refreshImages().catch((err) => setError(err instanceof Error ? err.message : "Failed to load images"));
  }, []);

  async function uploadFile(file: File) {
    const extension = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(extension)) {
      setError(`Unsupported file type ${extension || "(none)"}. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }

    setLoading("upload");
    setError(null);
    try {
      const uploaded = await uploadVisionImage(file, sourceType, equipmentHint || undefined);
      await refreshImages();
      setSelectedImageId(uploaded.id);
      setAnalysis(null);
      setSelectedItemKey(null);
      setQueryResult(null);
      setZoom(1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setLoading(null);
    }
  }

  function handleFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) uploadFile(file);
    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) uploadFile(file);
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(true);
  }

  function handleDragLeave() {
    setDragActive(false);
  }

  function selectImage(imageId: string) {
    setSelectedImageId(imageId);
    setAnalysis(null);
    setSelectedItemKey(null);
    setQueryResult(null);
    setZoom(1);
  }

  async function handleAnalyze() {
    if (!selectedImage) return;
    setLoading("analyze");
    setError(null);
    try {
      setAnalysis(await analyzeVisionImage(selectedImage.id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleQuery() {
    if (!query.trim()) return;
    setLoading("query");
    setError(null);
    try {
      const result = await multimodalQuery(query);
      setQueryResult(result.evidence);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleRunBenchmark(name: string) {
    setLoading(`benchmark:${name}`);
    setError(null);
    try {
      const report = await runBenchmark(name);
      setBenchmarkResults((prev) => ({ ...prev, [name]: report }));
    } catch (err) {
      setError(err instanceof Error ? err.message : `Benchmark ${name} failed`);
    } finally {
      setLoading(null);
    }
  }

  function toggleLayer(key: LayerKey) {
    setLayerVisibility((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  function selectItem(key: string) {
    setSelectedItemKey((prev) => (prev === key ? null : key));
  }

  function toggleExpanded(listKey: string) {
    setExpandedLists((prev) => ({ ...prev, [listKey]: !prev[listKey] }));
  }

  function visibleSlice<T>(items: T[], listKey: string): { visible: T[]; hiddenCount: number } {
    if (expandedLists[listKey] || items.length <= MAX_VISIBLE_LIST_ITEMS) {
      return { visible: items, hiddenCount: 0 };
    }
    return { visible: items.slice(0, MAX_VISIBLE_LIST_ITEMS), hiddenCount: items.length - MAX_VISIBLE_LIST_ITEMS };
  }

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[280px_1fr_320px]">
        {/* Component 2: Upload */}
        <div className="flex flex-col gap-4 rounded-lg border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-900">Upload Image</h2>

          <div className="flex flex-col gap-2">
            <label className="text-sm text-slate-600">
              Source type
              <select
                value={sourceType}
                onChange={(e) => setSourceType(e.target.value)}
                className="mt-1 w-full rounded border border-slate-300 px-3 py-1.5"
              >
                {VISION_SOURCE_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm text-slate-600">
              Equipment hint (optional)
              <input
                type="text"
                value={equipmentHint}
                onChange={(e) => setEquipmentHint(e.target.value)}
                placeholder="e.g. P-101"
                className="mt-1 w-full rounded border border-slate-300 px-3 py-1.5"
              />
            </label>
          </div>

          <div
            data-testid="vision-dropzone"
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed p-6 text-center text-sm text-slate-500 hover:border-indigo-400 ${
              dragActive ? "border-indigo-500 bg-indigo-50" : "border-slate-300"
            }`}
          >
            <span>{loading === "upload" ? "Uploading..." : "Drag & drop an image, or click to browse"}</span>
            <span className="text-xs text-slate-400">PNG, JPEG, BMP, TIFF</span>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_EXTENSIONS.join(",")}
              className="hidden"
              onChange={handleFileInputChange}
              disabled={loading === "upload"}
            />
          </div>

          <div className="flex flex-col gap-1">
            <h3 className="text-sm font-medium text-slate-700">Uploaded Images</h3>
            {images.length === 0 && <p className="text-sm text-slate-400">No images uploaded yet.</p>}
            <ul className="flex flex-col gap-1">
              {images.map((img) => {
                const isSelected = img.id === selectedImageId;
                const isAnalyzed = isSelected && analysis !== null;
                return (
                  <li key={img.id}>
                    <button
                      onClick={() => selectImage(img.id)}
                      className={`w-full rounded border px-2 py-1.5 text-left text-xs ${
                        isSelected ? "border-indigo-400 bg-indigo-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <img src={visionImageUrl(img.id)} alt="" className="mb-1 h-12 w-full rounded object-cover" />
                      <p className="truncate font-medium text-slate-800">{img.filename}</p>
                      <p className="text-slate-500">
                        {img.width}×{img.height} · {formatBytes(img.size_bytes)}
                      </p>
                      <p className="text-slate-500">{img.source_type}</p>
                      <span
                        className={`mt-1 inline-block rounded px-1 text-[10px] ${
                          isAnalyzed ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
                        }`}
                      >
                        {isAnalyzed ? "Analyzed" : "Not analyzed"}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        {/* Component 4: Interactive Viewer */}
        <div className="flex flex-col gap-3 rounded-lg border border-slate-200 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900">Interactive Viewer</h2>
            {selectedImage && (
              <div className="flex items-center gap-1 text-sm">
                <button
                  onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
                  className="rounded border border-slate-300 px-2 py-0.5"
                  aria-label="Zoom out"
                >
                  −
                </button>
                <span data-testid="zoom-level" className="w-12 text-center text-slate-600">
                  {Math.round(zoom * 100)}%
                </span>
                <button
                  onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
                  className="rounded border border-slate-300 px-2 py-0.5"
                  aria-label="Zoom in"
                >
                  +
                </button>
                <button onClick={() => setZoom(1)} className="ml-1 rounded border border-slate-300 px-2 py-0.5 text-xs">
                  Fit to screen
                </button>
              </div>
            )}
          </div>

          {selectedImage && (
            <div className="flex flex-wrap gap-3 text-xs">
              {(Object.keys(LAYER_LABELS) as LayerKey[]).map((key) => (
                <label key={key} className="flex items-center gap-1">
                  <input type="checkbox" checked={layerVisibility[key]} onChange={() => toggleLayer(key)} />
                  <span style={{ color: LAYER_COLORS[key] }}>■</span>
                  {LAYER_LABELS[key]}
                </label>
              ))}
            </div>
          )}

          {!selectedImage ? (
            <p className="flex h-64 items-center justify-center text-sm text-slate-400">
              Upload or select an image to begin.
            </p>
          ) : (
            <div className="relative overflow-auto rounded-lg border border-slate-200 bg-slate-100" style={{ maxHeight: 520 }}>
              <div className="relative" style={{ width: selectedImage.width * zoom, height: selectedImage.height * zoom }}>
                <img
                  src={visionImageUrl(selectedImage.id)}
                  alt={selectedImage.filename}
                  width={selectedImage.width * zoom}
                  height={selectedImage.height * zoom}
                  className="block"
                />
                {analysis && (
                  <svg
                    viewBox={`0 0 ${selectedImage.width} ${selectedImage.height}`}
                    preserveAspectRatio="none"
                    className="absolute inset-0 h-full w-full"
                  >
                    {layerVisibility.ocr &&
                      analysis.ocr_regions.map((region, i) => {
                        const key = `ocr-${i}`;
                        const [x1, y1, x2, y2] = region.bounding_box;
                        return (
                          <rect
                            key={key}
                            data-testid={key}
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke={LAYER_COLORS.ocr}
                            strokeWidth={selectedItemKey === key ? 4 : 2}
                            onClick={() => selectItem(key)}
                            style={{ cursor: "pointer" }}
                          >
                            <title>{`OCR: "${region.text}" (${Math.round(region.confidence * 100)}%)`}</title>
                          </rect>
                        );
                      })}
                    {layerVisibility.detection &&
                      analysis.detections.map((detection, i) => {
                        const key = `detection-${i}`;
                        const [x1, y1, x2, y2] = detection.bounding_box;
                        return (
                          <rect
                            key={key}
                            data-testid={key}
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke={LAYER_COLORS.detection}
                            strokeWidth={selectedItemKey === key ? 4 : 2}
                            onClick={() => selectItem(key)}
                            style={{ cursor: "pointer" }}
                          >
                            <title>{`Equipment: ${detection.equipment_type} (${Math.round(detection.confidence * 100)}%)`}</title>
                          </rect>
                        );
                      })}
                    {layerVisibility.diagram &&
                      analysis.diagram_symbols.map((symbol, i) => {
                        const key = `diagram-${i}`;
                        const [x1, y1, x2, y2] = symbol.bounding_box;
                        return (
                          <rect
                            key={key}
                            data-testid={key}
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke={LAYER_COLORS.diagram}
                            strokeWidth={selectedItemKey === key ? 4 : 2}
                            onClick={() => selectItem(key)}
                            style={{ cursor: "pointer" }}
                          >
                            <title>{`Symbol: ${symbol.symbol_type} (${Math.round(symbol.confidence * 100)}%)`}</title>
                          </rect>
                        );
                      })}
                    {layerVisibility.defect &&
                      analysis.defects.map((defect, i) => {
                        const key = `defect-${i}`;
                        const [x1, y1, x2, y2] = defect.bounding_box;
                        return (
                          <rect
                            key={key}
                            data-testid={key}
                            x={x1}
                            y={y1}
                            width={x2 - x1}
                            height={y2 - y1}
                            fill="none"
                            stroke={LAYER_COLORS.defect}
                            strokeWidth={selectedItemKey === key ? 4 : 2}
                            onClick={() => selectItem(key)}
                            style={{ cursor: "pointer" }}
                          >
                            <title>{`Defect: ${defect.defect_type} (${defect.severity}, ${Math.round(defect.confidence * 100)}%)`}</title>
                          </rect>
                        );
                      })}
                    {layerVisibility.gauge &&
                      analysis.gauge_readings.map((reading, i) => {
                        const key = `gauge-${i}`;
                        const [x1, y1, x2, y2] = reading.bounding_box;
                        const cx = (x1 + x2) / 2;
                        const cy = (y1 + y2) / 2;
                        const radius = Math.max(1, (x2 - x1) / 2);
                        const angle = reading.needle_angle_degrees ?? 0;
                        const rad = (angle * Math.PI) / 180;
                        const nx = cx + radius * Math.sin(rad);
                        const ny = cy - radius * Math.cos(rad);
                        return (
                          <g key={key} data-testid={key} onClick={() => selectItem(key)} style={{ cursor: "pointer" }}>
                            <circle
                              cx={cx}
                              cy={cy}
                              r={radius}
                              fill="none"
                              stroke={LAYER_COLORS.gauge}
                              strokeWidth={selectedItemKey === key ? 4 : 2}
                            />
                            {reading.needle_angle_degrees !== null && (
                              <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={LAYER_COLORS.gauge} strokeWidth={2} />
                            )}
                            <title>
                              {`Gauge: ${
                                reading.normalized_value !== null ? `${Math.round(reading.normalized_value * 100)}% of scale` : "unreadable"
                              } (${Math.round(reading.confidence * 100)}% confidence)`}
                            </title>
                          </g>
                        );
                      })}
                  </svg>
                )}
              </div>
            </div>
          )}

          <div className="flex items-center justify-between">
            <button
              onClick={handleAnalyze}
              disabled={!selectedImage || loading === "analyze"}
              className="rounded-lg bg-indigo-600 px-4 py-1.5 font-medium text-white disabled:opacity-50"
            >
              {loading === "analyze" ? "Analyzing..." : "Run Analysis"}
            </button>
            {analysis && (
              <ul className="text-xs text-emerald-700">
                <li>
                  ✓ OCR complete — {analysis.ocr_regions.length} region(s) · ✓ Detection complete — {analysis.detections.length} object(s)
                  · ✓ Diagram parsing complete — {analysis.diagram_symbols.length} symbol(s) · ✓ Gauge reading complete —{" "}
                  {analysis.gauge_readings.length} gauge(s) · ✓ Defect detection complete — {analysis.defects.length} defect(s) · ✓ Graph
                  integration complete
                </li>
              </ul>
            )}
          </div>
        </div>

        {/* Component 3/5/6/7: Analysis Results */}
        <div className="flex flex-col gap-4 rounded-lg border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-900">Analysis Results</h2>
          {!analysis ? (
            <p className="text-sm text-slate-400">Run analysis to see OCR, detections, diagram symbols, gauges, and defects.</p>
          ) : (
            <>
              <ResultSection title="OCR Regions" listKey="ocr" items={analysis.ocr_regions} visibleSlice={visibleSlice} onToggleExpand={toggleExpanded}>
                {(region, i) => {
                  const key = `ocr-${i}`;
                  return (
                    <button
                      key={key}
                      onClick={() => selectItem(key)}
                      className={`w-full rounded border px-2 py-1 text-left text-xs ${
                        selectedItemKey === key ? "border-emerald-400 bg-emerald-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <p className="truncate font-medium text-slate-800">"{region.text}"</p>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(region.confidence)}`}>
                        {Math.round(region.confidence * 100)}%
                      </span>
                    </button>
                  );
                }}
              </ResultSection>

              <ResultSection title="Detections" listKey="detection" items={analysis.detections} visibleSlice={visibleSlice} onToggleExpand={toggleExpanded}>
                {(detection, i) => {
                  const key = `detection-${i}`;
                  return (
                    <button
                      key={key}
                      onClick={() => selectItem(key)}
                      className={`w-full rounded border px-2 py-1 text-left text-xs ${
                        selectedItemKey === key ? "border-indigo-400 bg-indigo-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <p className="font-medium text-slate-800">{detection.equipment_type}</p>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(detection.confidence)}`}>
                        {Math.round(detection.confidence * 100)}%
                      </span>
                    </button>
                  );
                }}
              </ResultSection>

              <ResultSection title="Diagram Symbols" listKey="diagram" items={analysis.diagram_symbols} visibleSlice={visibleSlice} onToggleExpand={toggleExpanded}>
                {(symbol, i) => {
                  const key = `diagram-${i}`;
                  return (
                    <button
                      key={key}
                      onClick={() => selectItem(key)}
                      className={`w-full rounded border px-2 py-1 text-left text-xs ${
                        selectedItemKey === key ? "border-violet-400 bg-violet-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <p className="font-medium text-slate-800">{symbol.symbol_type}</p>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(symbol.confidence)}`}>
                        {Math.round(symbol.confidence * 100)}%
                      </span>
                    </button>
                  );
                }}
              </ResultSection>

              <ResultSection title="Gauge Readings" listKey="gauge" items={analysis.gauge_readings} visibleSlice={visibleSlice} onToggleExpand={toggleExpanded}>
                {(reading, i) => {
                  const key = `gauge-${i}`;
                  return (
                    <button
                      key={key}
                      onClick={() => selectItem(key)}
                      className={`w-full rounded border px-2 py-1 text-left text-xs ${
                        selectedItemKey === key ? "border-amber-400 bg-amber-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <p className="font-medium text-slate-800">
                        {reading.valid && reading.normalized_value !== null
                          ? `${Math.round(reading.normalized_value * 100)}% of scale (uncalibrated)`
                          : "Unreadable"}
                      </p>
                      <p className="text-slate-500">
                        needle: {reading.needle_angle_degrees !== null ? `${reading.needle_angle_degrees.toFixed(1)}°` : "n/a"} · raw
                        reading: {reading.needle_angle_degrees ?? "n/a"} · validated: {reading.valid ? "yes" : "no"} (
                        {reading.validation_note})
                      </p>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(reading.confidence)}`}>
                        {Math.round(reading.confidence * 100)}%
                      </span>
                    </button>
                  );
                }}
              </ResultSection>

              <ResultSection title="Defects" listKey="defect" items={analysis.defects} visibleSlice={visibleSlice} onToggleExpand={toggleExpanded}>
                {(defect, i) => {
                  const key = `defect-${i}`;
                  return (
                    <button
                      key={key}
                      onClick={() => selectItem(key)}
                      className={`w-full rounded border px-2 py-1 text-left text-xs ${
                        selectedItemKey === key ? "border-red-400 bg-red-50" : "border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      <p className="font-medium text-slate-800">
                        {defect.defect_type} <span className="uppercase text-slate-500">({defect.severity})</span>
                      </p>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(defect.confidence)}`}>
                        {Math.round(defect.confidence * 100)}%
                      </span>
                    </button>
                  );
                }}
              </ResultSection>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Component 8: Knowledge Graph Panel */}
        <div className="rounded-lg border border-slate-200 p-4">
          <h2 className="mb-2 text-lg font-semibold text-slate-900">Knowledge Graph</h2>
          {!analysis ? (
            <p className="text-sm text-slate-400">Run analysis to see the graph entities created from this image.</p>
          ) : (
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-slate-500">
                  <th className="pb-1">Type</th>
                  <th className="pb-1">Label</th>
                  <th className="pb-1">Confidence</th>
                  <th className="pb-1">Graph Node</th>
                </tr>
              </thead>
              <tbody>
                {analysis.ocr_regions.map((r, i) => (
                  <GraphRow
                    key={`ocr-${i}`}
                    itemKey={`ocr-${i}`}
                    type="OCR Region"
                    label={r.text}
                    confidence={r.confidence}
                    nodeId={r.graph_node_id}
                    selected={selectedItemKey === `ocr-${i}`}
                    onSelect={selectItem}
                  />
                ))}
                {analysis.detections.map((d, i) => (
                  <GraphRow
                    key={`detection-${i}`}
                    itemKey={`detection-${i}`}
                    type="Detected Object"
                    label={d.equipment_type}
                    confidence={d.confidence}
                    nodeId={d.graph_node_id}
                    selected={selectedItemKey === `detection-${i}`}
                    onSelect={selectItem}
                  />
                ))}
                {analysis.diagram_symbols.map((s, i) => (
                  <GraphRow
                    key={`diagram-${i}`}
                    itemKey={`diagram-${i}`}
                    type="Diagram Symbol"
                    label={s.symbol_type}
                    confidence={s.confidence}
                    nodeId={s.graph_node_id}
                    selected={selectedItemKey === `diagram-${i}`}
                    onSelect={selectItem}
                  />
                ))}
                {analysis.gauge_readings.map((g, i) => (
                  <GraphRow
                    key={`gauge-${i}`}
                    itemKey={`gauge-${i}`}
                    type="Gauge Reading"
                    label={g.normalized_value !== null ? `${Math.round(g.normalized_value * 100)}%` : "unreadable"}
                    confidence={g.confidence}
                    nodeId={g.graph_node_id}
                    selected={selectedItemKey === `gauge-${i}`}
                    onSelect={selectItem}
                  />
                ))}
                {analysis.defects.map((d, i) => (
                  <GraphRow
                    key={`defect-${i}`}
                    itemKey={`defect-${i}`}
                    type="Defect"
                    label={d.defect_type}
                    confidence={d.confidence}
                    nodeId={d.graph_node_id}
                    selected={selectedItemKey === `defect-${i}`}
                    onSelect={selectItem}
                  />
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Component 9: Multimodal Query */}
        <div className="rounded-lg border border-slate-200 p-4">
          <h2 className="mb-2 text-lg font-semibold text-slate-900">Multimodal Query</h2>
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='e.g. "What is the gauge reading?"'
              className="flex-1 rounded border border-slate-300 px-3 py-1.5 text-sm"
              onKeyDown={(e) => {
                if (e.key === "Enter") handleQuery();
              }}
            />
            <button
              onClick={handleQuery}
              disabled={loading === "query" || !query.trim()}
              className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading === "query" ? "Asking..." : "Ask"}
            </button>
          </div>
          {queryResult && (
            <ul className="mt-3 flex flex-col gap-2">
              {queryResult.length === 0 && <p className="text-sm text-slate-400">No evidence found for this query.</p>}
              {queryResult.map((evidence, i) => {
                const badge = evidenceSourceBadge(evidence.source);
                return (
                  <li key={i} className="rounded border border-slate-200 p-2 text-sm">
                    <div className="mb-1 flex items-center gap-2">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ${badge.className}`}>{badge.label}</span>
                      <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(evidence.confidence)}`}>
                        {Math.round(evidence.confidence * 100)}%
                      </span>
                    </div>
                    <p className="text-slate-700">{evidence.text}</p>
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      {/* Component 10: Research Integration */}
      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Vision Research Benchmarks</h2>
        <div className="flex flex-wrap gap-2">
          {VISION_BENCHMARKS.map((name) => (
            <button
              key={name}
              onClick={() => handleRunBenchmark(name)}
              disabled={loading === `benchmark:${name}`}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {loading === `benchmark:${name}` ? "Running..." : name}
            </button>
          ))}
        </div>
        {Object.keys(benchmarkResults).length > 0 && (
          <table className="mt-3 w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Benchmark</th>
                <th className="pb-1">Metrics</th>
                <th className="pb-1">Latency</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(benchmarkResults).map(([name, report]) => (
                <tr key={name} className="border-t border-slate-100">
                  <td className="py-1 font-medium">{name}</td>
                  <td className="py-1 text-slate-600">
                    {report.metrics.map((m) => `${m.name}=${m.value}`).join(", ")}
                  </td>
                  <td className="py-1">{report.duration_seconds.toFixed(3)}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function ResultSection<T>({
  title,
  listKey,
  items,
  visibleSlice,
  onToggleExpand,
  children,
}: {
  title: string;
  listKey: string;
  items: T[];
  visibleSlice: (items: T[], listKey: string) => { visible: T[]; hiddenCount: number };
  onToggleExpand: (listKey: string) => void;
  children: (item: T, index: number) => ReactNode;
}) {
  if (items.length === 0) return null;
  const { visible, hiddenCount } = visibleSlice(items, listKey);
  return (
    <div>
      <h3 className="mb-1 text-sm font-medium text-slate-700">
        {title} ({items.length})
      </h3>
      <div className="flex flex-col gap-1">{visible.map((item, i) => children(item, i))}</div>
      {hiddenCount > 0 && (
        <button onClick={() => onToggleExpand(listKey)} className="mt-1 text-xs text-indigo-600 hover:underline">
          Show {hiddenCount} more
        </button>
      )}
    </div>
  );
}

function GraphRow({
  itemKey,
  type,
  label,
  confidence,
  nodeId,
  selected,
  onSelect,
}: {
  itemKey: string;
  type: string;
  label: string;
  confidence: number;
  nodeId: string | null;
  selected: boolean;
  onSelect: (key: string) => void;
}) {
  return (
    <tr className={`cursor-pointer border-t border-slate-100 ${selected ? "bg-indigo-50" : "hover:bg-slate-50"}`} onClick={() => onSelect(itemKey)}>
      <td className="py-1">{type}</td>
      <td className="py-1 font-medium">{label}</td>
      <td className="py-1">
        <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(confidence)}`}>{Math.round(confidence * 100)}%</span>
      </td>
      <td className="py-1 font-mono text-xs text-slate-500">{nodeId ? nodeId.slice(0, 8) : "—"}</td>
    </tr>
  );
}

export function VisionPage() {
  return (
    <VisionErrorBoundary>
      <VisionWorkspace />
    </VisionErrorBoundary>
  );
}

export { formatBytes };
