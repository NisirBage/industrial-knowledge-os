import type { ElementDefinition, Stylesheet } from "cytoscape";
import { Component, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CytoscapeGraph } from "../components/CytoscapeGraph";
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
  listModels,
  runBenchmark,
  simulatePropagation,
  trainGraphEmbeddings,
  trainNodeClassifier,
} from "../services/api";
import type {
  BenchmarkReport,
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

const PREDICTION_TASK_TYPES = [
  "failure_risk",
  "remaining_useful_life",
  "inspection_priority",
  "maintenance_priority",
  "recommendation_priority",
  "missing_documentation_risk",
  "compliance_risk",
] as const;

const COMMUNITY_PALETTE = ["#4f46e5", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#65a30d"];

function communityColor(communityId: number | null): string {
  if (communityId === null) return "#94a3b8";
  return COMMUNITY_PALETTE[communityId % COMMUNITY_PALETTE.length];
}

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.7) return "bg-emerald-50 text-emerald-700";
  if (confidence >= 0.4) return "bg-amber-50 text-amber-700";
  return "bg-red-50 text-red-700";
}

interface ErrorBoundaryProps {
  children: ReactNode;
}
interface ErrorBoundaryState {
  error: Error | null;
}

class GraphIntelligenceErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="mx-auto max-w-4xl p-6">
          <p className="rounded-lg bg-red-50 p-3 text-red-700">
            The Graph Intelligence workspace crashed: {this.state.error.message}. Try reloading the page.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

function GraphIntelligenceWorkspace() {
  const [summary, setSummary] = useState<GraphSummary | null>(null);
  const [graphExport, setGraphExport] = useState<GraphExport | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [typeFilter, setTypeFilter] = useState<Record<string, boolean>>({});
  const [nameFilter, setNameFilter] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [highlightedNodeIds, setHighlightedNodeIds] = useState<Set<string> | null>(null);

  const [equipmentQuery, setEquipmentQuery] = useState("");
  const [nodeFeatures, setNodeFeatures] = useState<NodeFeatures | null>(null);

  const [embeddingModel, setEmbeddingModel] = useState<string>(GRAPH_EMBEDDING_MODELS[0]);
  const [embeddingDimensions, setEmbeddingDimensions] = useState(32);
  const [embeddingResult, setEmbeddingResult] = useState<EmbeddingTrainResult | null>(null);

  const [similarityTarget, setSimilarityTarget] = useState("");
  const [similarNodes, setSimilarNodes] = useState<SimilarNode[] | null>(null);
  const [explanation, setExplanation] = useState<GraphExplanation | null>(null);

  const [linkTarget, setLinkTarget] = useState("");
  const [linkCandidates, setLinkCandidates] = useState<LinkPredictionCandidate[] | null>(null);

  const [propagationTarget, setPropagationTarget] = useState("");
  const [propagationType, setPropagationType] = useState<string>(PROPAGATION_TYPES[0]);
  const [propagationDepth, setPropagationDepth] = useState(3);
  const [propagationResult, setPropagationResult] = useState<PropagationResult | null>(null);

  const [classificationTaskType, setClassificationTaskType] = useState<string>(PREDICTION_TASK_TYPES[0]);
  const [classificationEquipment, setClassificationEquipment] = useState("");
  const [classificationResult, setClassificationResult] = useState<NodeClassificationResult | null>(null);

  const [registeredModels, setRegisteredModels] = useState<ModelRecord[]>([]);
  const [benchmarkResults, setBenchmarkResults] = useState<Record<string, BenchmarkReport>>({});

  async function refreshGraph() {
    const [nextSummary, nextExport] = await Promise.all([getGraphSummary(), getGraphExport()]);
    setSummary(nextSummary);
    setGraphExport(nextExport);
    setTypeFilter((prev) => {
      const next = { ...prev };
      for (const type of nextSummary.node_types) {
        if (!(type in next)) next[type] = true;
      }
      return next;
    });
  }

  useEffect(() => {
    refreshGraph().catch((err) => setError(err instanceof Error ? err.message : "Failed to load graph"));
  }, []);

  const nodeById = useMemo(() => {
    const map = new Map<string, GraphExport["nodes"][number]>();
    graphExport?.nodes.forEach((n) => map.set(n.id, n));
    return map;
  }, [graphExport]);

  const visibleElements = useMemo<ElementDefinition[]>(() => {
    if (!graphExport) return [];
    const nameNeedle = nameFilter.trim().toLowerCase();
    const visibleNodeIds = new Set(
      graphExport.nodes
        .filter((n) => typeFilter[n.type] !== false)
        .filter((n) => !nameNeedle || n.canonical_name.toLowerCase().includes(nameNeedle))
        .map((n) => n.id),
    );
    const nodeElements: ElementDefinition[] = graphExport.nodes
      .filter((n) => visibleNodeIds.has(n.id))
      .map((n) => ({
        data: { id: n.id, label: n.canonical_name, type: n.type, community_id: n.community_id },
      }));
    const edgeElements: ElementDefinition[] = graphExport.edges
      .filter((e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target))
      .map((e, i) => ({
        data: { id: `edge-${i}`, source: e.source, target: e.target, label: e.relation_type },
      }));
    return [...nodeElements, ...edgeElements];
  }, [graphExport, typeFilter, nameFilter]);

  const graphStylesheet = useMemo<Stylesheet[]>(
    () => [
      {
        selector: "node",
        style: {
          "background-color": (ele) => communityColor(ele.data("community_id")),
          label: "data(label)",
          "font-size": 8,
          width: 24,
          height: 24,
          color: "#334155",
          "text-valign": "bottom",
          "text-margin-y": 4,
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.5,
          "line-color": "#cbd5e1",
          "target-arrow-color": "#cbd5e1",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          "font-size": 6,
        },
      },
      {
        selector: "node.selected",
        style: { "border-width": 3, "border-color": "#4f46e5" },
      },
      {
        selector: "node.highlighted",
        style: { "background-color": "#dc2626", "border-width": 3, "border-color": "#dc2626" },
      },
    ],
    [],
  );

  function toggleType(type: string) {
    setTypeFilter((prev) => ({ ...prev, [type]: !prev[type] }));
  }

  async function handleLookupFeatures() {
    if (!equipmentQuery.trim()) return;
    setLoading("features");
    setError(null);
    try {
      const features = await getNodeFeatures(equipmentQuery.trim());
      setNodeFeatures(features);
      setSelectedNodeId(features.node_id);
      setHighlightedNodeIds(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Feature lookup failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleTrainEmbeddings() {
    setLoading("embeddings");
    setError(null);
    try {
      const result = await trainGraphEmbeddings(embeddingModel, embeddingDimensions);
      setEmbeddingResult(result);
      await refreshGraph();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Embedding training failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleSimilaritySearch() {
    if (!similarityTarget.trim()) return;
    setLoading("similarity");
    setError(null);
    try {
      const [neighbors, nodeExplanation] = await Promise.all([
        getSimilarNodes(similarityTarget.trim()),
        explainNode(similarityTarget.trim()),
      ]);
      setSimilarNodes(neighbors);
      setExplanation(nodeExplanation);
      setSelectedNodeId(nodeExplanation.node_id);
      setHighlightedNodeIds(new Set(neighbors.map((n) => n.node_id)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Similarity search failed (train embeddings first)");
    } finally {
      setLoading(null);
    }
  }

  async function handleLinkPrediction() {
    if (!linkTarget.trim()) return;
    setLoading("link-prediction");
    setError(null);
    try {
      const candidates = await getLinkPredictionCandidates(linkTarget.trim());
      setLinkCandidates(candidates);
      setHighlightedNodeIds(new Set(candidates.map((c) => c.target_id)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Link prediction failed (train embeddings first)");
    } finally {
      setLoading(null);
    }
  }

  async function handlePropagation() {
    if (!propagationTarget.trim()) return;
    setLoading("propagation");
    setError(null);
    try {
      const result = await simulatePropagation(propagationTarget.trim(), propagationType, propagationDepth);
      setPropagationResult(result);
      setSelectedNodeId(result.source_node_id);
      setHighlightedNodeIds(new Set(result.steps.map((s) => s.node_id)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Propagation simulation failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleTrainClassifier() {
    const names = classificationEquipment
      .split(",")
      .map((n) => n.trim())
      .filter(Boolean);
    if (names.length === 0) return;
    setLoading("classification");
    setError(null);
    try {
      const result = await trainNodeClassifier(classificationTaskType, names);
      setClassificationResult(result);
      if (result.registered) {
        setRegisteredModels(await listModels(classificationTaskType));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Node classification training failed (train embeddings first)");
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

  const selectedNode = selectedNodeId ? nodeById.get(selectedNodeId) ?? null : null;

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {/* Component 1: Graph summary */}
      {summary && (
        <div className="flex flex-wrap gap-4 rounded-lg border border-slate-200 p-4 text-sm">
          <span>
            <strong>{summary.node_count}</strong> nodes
          </span>
          <span>
            <strong>{summary.edge_count}</strong> edges
          </span>
          <span className="text-slate-500">Types: {summary.node_types.join(", ") || "none"}</span>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[240px_1fr_320px]">
        {/* Component 2: Filters */}
        <div className="flex flex-col gap-4 rounded-lg border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-900">Filters</h2>
          <label className="text-sm text-slate-600">
            Search by name
            <input
              type="text"
              value={nameFilter}
              onChange={(e) => setNameFilter(e.target.value)}
              placeholder="e.g. P-101"
              className="mt-1 w-full rounded border border-slate-300 px-3 py-1.5"
            />
          </label>
          <div className="flex flex-col gap-1">
            <h3 className="text-sm font-medium text-slate-700">Node types</h3>
            {Object.keys(typeFilter).map((type) => (
              <label key={type} className="flex items-center gap-2 text-xs">
                <input type="checkbox" checked={typeFilter[type] !== false} onChange={() => toggleType(type)} />
                <span style={{ color: communityColor(null) }}>■</span>
                {type}
              </label>
            ))}
          </div>

          <div className="flex flex-col gap-2 border-t border-slate-100 pt-3">
            <h3 className="text-sm font-medium text-slate-700">Node Feature Lookup</h3>
            <input
              type="text"
              value={equipmentQuery}
              onChange={(e) => setEquipmentQuery(e.target.value)}
              placeholder="Equipment name"
              className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleLookupFeatures()}
            />
            <button
              onClick={handleLookupFeatures}
              disabled={loading === "features"}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading === "features" ? "Looking up..." : "Look up features"}
            </button>
            {nodeFeatures && (
              <dl className="mt-1 grid grid-cols-2 gap-x-2 gap-y-0.5 text-xs text-slate-600">
                <dt>Degree</dt>
                <dd>{nodeFeatures.degree}</dd>
                <dt>PageRank</dt>
                <dd>{nodeFeatures.pagerank.toFixed(4)}</dd>
                <dt>Betweenness</dt>
                <dd>{nodeFeatures.betweenness.toFixed(4)}</dd>
                <dt>Community</dt>
                <dd>{nodeFeatures.community_id ?? "—"}</dd>
                <dt>Maintenance</dt>
                <dd>{nodeFeatures.maintenance_count}</dd>
                <dt>Incidents</dt>
                <dd>{nodeFeatures.incident_count}</dd>
                <dt>Defects</dt>
                <dd>{nodeFeatures.defect_count}</dd>
              </dl>
            )}
          </div>
        </div>

        {/* Component 1: Interactive graph */}
        <div className="flex flex-col gap-3 rounded-lg border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-900">Knowledge Graph</h2>
          <CytoscapeGraph
            elements={visibleElements}
            stylesheet={graphStylesheet}
            onNodeTap={setSelectedNodeId}
            selectedNodeId={selectedNodeId}
            highlightedNodeIds={highlightedNodeIds}
          />
          {selectedNode && (
            <p className="text-xs text-slate-600">
              Selected: <span className="font-medium">{selectedNode.canonical_name}</span> ({selectedNode.type}) — community{" "}
              {selectedNode.community_id ?? "—"}
            </p>
          )}
        </div>

        {/* Component 3: Embedding training */}
        <div className="flex flex-col gap-3 rounded-lg border border-slate-200 p-4">
          <h2 className="text-lg font-semibold text-slate-900">Train Embeddings</h2>
          <label className="text-sm text-slate-600">
            Model
            <select
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-1.5"
            >
              {GRAPH_EMBEDDING_MODELS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm text-slate-600">
            Dimensions
            <input
              type="number"
              value={embeddingDimensions}
              min={4}
              max={128}
              onChange={(e) => setEmbeddingDimensions(Number(e.target.value))}
              className="mt-1 w-full rounded border border-slate-300 px-3 py-1.5"
            />
          </label>
          <button
            onClick={handleTrainEmbeddings}
            disabled={loading === "embeddings"}
            className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading === "embeddings" ? "Training..." : "Train"}
          </button>
          {embeddingResult && (
            <p className="text-xs text-emerald-700">
              ✓ Trained {embeddingResult.model_name} ({embeddingResult.dimensions}d) on {embeddingResult.node_count} nodes in{" "}
              {embeddingResult.training_seconds.toFixed(3)}s
            </p>
          )}

          <div className="flex flex-col gap-2 border-t border-slate-100 pt-3">
            <h3 className="text-sm font-medium text-slate-700">Similarity Search &amp; Explanation</h3>
            <input
              type="text"
              value={similarityTarget}
              onChange={(e) => setSimilarityTarget(e.target.value)}
              placeholder="Equipment name"
              className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleSimilaritySearch()}
            />
            <button
              onClick={handleSimilaritySearch}
              disabled={loading === "similarity"}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {loading === "similarity" ? "Searching..." : "Find similar nodes"}
            </button>
            {similarNodes && similarNodes.length > 0 && (
              <div style={{ width: "100%", height: 160 }}>
                <ResponsiveContainer>
                  <BarChart data={similarNodes.map((n) => ({ name: n.canonical_name, score: n.score }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-30} textAnchor="end" height={50} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} />
                    <Tooltip />
                    <Bar dataKey="score" fill="#4f46e5" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
            {explanation && <p className="text-xs text-slate-600">{explanation.text}</p>}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Component 5: Link prediction explorer */}
        <div className="rounded-lg border border-slate-200 p-4">
          <h2 className="mb-2 text-lg font-semibold text-slate-900">Link Prediction</h2>
          <div className="flex gap-2">
            <input
              type="text"
              value={linkTarget}
              onChange={(e) => setLinkTarget(e.target.value)}
              placeholder="Equipment name"
              className="flex-1 rounded border border-slate-300 px-3 py-1.5 text-sm"
              onKeyDown={(e) => e.key === "Enter" && handleLinkPrediction()}
            />
            <button
              onClick={handleLinkPrediction}
              disabled={loading === "link-prediction"}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading === "link-prediction" ? "Predicting..." : "Predict"}
            </button>
          </div>
          {linkCandidates && (
            <ul className="mt-3 flex flex-col gap-1">
              {linkCandidates.length === 0 && <p className="text-sm text-slate-400">No candidates found.</p>}
              {linkCandidates.map((c, i) => (
                <li key={i} className="rounded border border-slate-200 p-2 text-xs">
                  <p className="font-medium text-slate-800">{c.target_canonical_name}</p>
                  <p className="text-slate-500">{c.explanation}</p>
                  <span className={`rounded px-1 text-[10px] ${confidenceBadgeClass(c.confidence)}`}>
                    {Math.round(c.confidence * 100)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Component 7: Risk propagation */}
        <div className="rounded-lg border border-slate-200 p-4">
          <h2 className="mb-2 text-lg font-semibold text-slate-900">Risk Propagation</h2>
          <div className="flex flex-col gap-2">
            <input
              type="text"
              value={propagationTarget}
              onChange={(e) => setPropagationTarget(e.target.value)}
              placeholder="Equipment name"
              className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
            />
            <div className="flex gap-2">
              <select
                value={propagationType}
                onChange={(e) => setPropagationType(e.target.value)}
                className="flex-1 rounded border border-slate-300 px-2 py-1.5 text-sm"
              >
                {PROPAGATION_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
              <input
                type="number"
                value={propagationDepth}
                min={1}
                max={6}
                onChange={(e) => setPropagationDepth(Number(e.target.value))}
                className="w-16 rounded border border-slate-300 px-2 py-1.5 text-sm"
              />
            </div>
            <button
              onClick={handlePropagation}
              disabled={loading === "propagation"}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading === "propagation" ? "Simulating..." : "Simulate"}
            </button>
          </div>
          {propagationResult && (
            <ul className="mt-3 flex flex-col gap-1">
              {propagationResult.steps.map((s, i) => (
                <li key={i} className="rounded border border-slate-200 p-2 text-xs">
                  <p className="font-medium text-slate-800">
                    {s.canonical_name} (depth {s.depth})
                  </p>
                  <p className="text-slate-500">risk: {s.propagated_risk.toFixed(3)} · path: {s.path.join(" → ")}</p>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Component 4/6/10: Node classification + ML Lab model comparison */}
        <div className="rounded-lg border border-slate-200 p-4">
          <h2 className="mb-2 text-lg font-semibold text-slate-900">Node Classification</h2>
          <div className="flex flex-col gap-2">
            <select
              value={classificationTaskType}
              onChange={(e) => setClassificationTaskType(e.target.value)}
              className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm"
            >
              {PREDICTION_TASK_TYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <input
              type="text"
              value={classificationEquipment}
              onChange={(e) => setClassificationEquipment(e.target.value)}
              placeholder="Equipment names, comma-separated"
              className="w-full rounded border border-slate-300 px-3 py-1.5 text-sm"
            />
            <button
              onClick={handleTrainClassifier}
              disabled={loading === "classification"}
              className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              {loading === "classification" ? "Training..." : "Train vs. rule baseline"}
            </button>
          </div>
          {classificationResult && (
            <div className="mt-2 text-xs">
              {classificationResult.registered ? (
                <p className="text-emerald-700">
                  ✓ Registered {classificationResult.algorithm} for {classificationResult.task_type} (n=
                  {classificationResult.sample_size})
                </p>
              ) : (
                <p className="text-amber-700">Not enough resolvable equipment to train (need at least 2).</p>
              )}
              {classificationResult.feature_importance.length > 0 && (
                <div style={{ width: "100%", height: 120 }} className="mt-2">
                  <ResponsiveContainer>
                    <BarChart data={classificationResult.feature_importance.slice(0, 8)}>
                      <XAxis dataKey="feature_name" tick={{ fontSize: 8 }} />
                      <YAxis tick={{ fontSize: 8 }} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#059669" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}
          {registeredModels.length > 0 && (
            <table className="mt-3 w-full text-left text-xs">
              <thead>
                <tr className="text-slate-500">
                  <th className="pb-1">Algorithm</th>
                  <th className="pb-1">Metric</th>
                </tr>
              </thead>
              <tbody>
                {registeredModels.map((m) => (
                  <tr key={m.model_id} className="border-t border-slate-100">
                    <td className="py-1">
                      {m.algorithm.startsWith("graph_") ? (
                        <span className="rounded bg-violet-50 px-1 text-violet-700">Graph</span>
                      ) : (
                        <span className="rounded bg-slate-100 px-1 text-slate-600">Classical</span>
                      )}{" "}
                      {m.algorithm}
                    </td>
                    <td className="py-1">{Object.entries(m.metrics).slice(0, 1).map(([k, v]) => `${k}=${v}`)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* Component 9: Research Integration */}
      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Graph Research Benchmarks</h2>
        <div className="flex flex-wrap gap-2">
          {GRAPH_BENCHMARKS.map((name) => (
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
                  <td className="py-1 text-slate-600">{report.metrics.map((m) => `${m.name}=${m.value}`).join(", ")}</td>
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

export function GraphIntelligencePage() {
  return (
    <GraphIntelligenceErrorBoundary>
      <GraphIntelligenceWorkspace />
    </GraphIntelligenceErrorBoundary>
  );
}
