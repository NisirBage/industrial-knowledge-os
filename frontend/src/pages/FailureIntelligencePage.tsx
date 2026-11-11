import type { ElementDefinition, NodeSingular, StylesheetJsonBlock as Stylesheet } from "cytoscape";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CytoscapeGraph } from "../components/CytoscapeGraph";
import { ExecutionSummaryCard } from "../components/ExecutionSummaryCard";
import {
  COUNTERFACTUAL_TOGGLES,
  getExecutionStatistics,
  getOptimizationRecommendations,
  getPatternAttribution,
  getRootCauseGraph,
  listPatterns,
  minePatterns,
  runBenchmark,
  simulateCounterfactual,
} from "../services/api";
import type {
  BenchmarkReport,
  CounterfactualResult,
  ExecutionStatistics,
  FailureAttribution,
  FailurePattern,
  OptimizationRecommendation,
  RootCauseEdge,
  RootCauseGraph,
  RootCauseNode,
} from "../services/api";

const NODE_TYPE_COLORS: Record<string, string> = {
  execution: "#4f46e5",
  failure: "#dc2626",
  component: "#94a3b8",
  claim: "#d97706",
  recommendation: "#059669",
  evidence: "#0891b2",
  pattern: "#7c3aed",
};

function severityBadgeClass(severity: string): string {
  if (severity === "critical") return "bg-red-100 text-red-800";
  if (severity === "high") return "bg-red-50 text-red-700";
  if (severity === "medium") return "bg-amber-50 text-amber-700";
  return "bg-slate-100 text-slate-600";
}

function findDominantCause(graph: RootCauseGraph, failureNodeId: string): RootCauseNode | null {
  const causedByEdges = graph.edges.filter((e) => e.source === failureNodeId && e.type === "caused_by");
  if (causedByEdges.length === 0) return null;
  const best = causedByEdges.reduce((a, b) => (b.confidence > a.confidence ? b : a));
  return graph.nodes.find((n) => n.id === best.target) ?? null;
}

function computeDependencyDepth(graph: RootCauseGraph, nodeId: string, visited: Set<string> = new Set()): number {
  if (visited.has(nodeId)) return 0;
  const nextVisited = new Set(visited).add(nodeId);
  const causal = graph.edges.filter((e) => e.source === nodeId && (e.type === "caused_by" || e.type === "contributed_to"));
  if (causal.length === 0) return 0;
  return 1 + Math.max(...causal.map((e) => computeDependencyDepth(graph, e.target, nextVisited)));
}

function findCriticalPath(edges: RootCauseEdge[], fromId: string, toId: string): string[] | null {
  if (fromId === toId) return [fromId];
  const visited = new Set([fromId]);
  const queue: string[][] = [[fromId]];
  while (queue.length > 0) {
    const path = queue.shift();
    if (!path) break;
    const current = path[path.length - 1];
    for (const edge of edges.filter((e) => e.source === current)) {
      if (edge.target === toId) return [...path, toId];
      if (!visited.has(edge.target)) {
        visited.add(edge.target);
        queue.push([...path, edge.target]);
      }
    }
  }
  return null;
}

function DashboardMetrics() {
  const [statistics, setStatistics] = useState<ExecutionStatistics | null>(null);
  const [patterns, setPatterns] = useState<FailurePattern[]>([]);
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [overheadReport, setOverheadReport] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getExecutionStatistics(), listPatterns(), getOptimizationRecommendations(), runBenchmark("execution_trace_overhead")])
      .then(([stats, pats, recs, report]) => {
        setStatistics(stats);
        setPatterns(pats);
        setRecommendations(recs);
        setOverheadReport(report);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load dashboard metrics"));
  }, []);

  const failureCount = useMemo(() => new Set(patterns.flatMap((p) => p.example_execution_ids)).size, [patterns]);
  const mostFrequent = useMemo(() => (patterns.length === 0 ? null : patterns.reduce((a, b) => (b.support > a.support ? b : a))), [patterns]);
  const cacheHitRatio = overheadReport?.metrics.find((m) => m.name === "cache_hit_ratio")?.value ?? null;
  const avgLatency = statistics
    ? Object.values(statistics.average_latency_by_component).reduce((a, b) => a + b, 0) / Math.max(1, Object.keys(statistics.average_latency_by_component).length)
    : null;

  const trendData = useMemo(() => patterns.map((p) => ({ category: p.category, support: p.support })), [patterns]);

  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Dashboard Metrics</h2>
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
        {[
          ["Execution Count", statistics?.trace_count ?? "—"],
          ["Failure Count", failureCount],
          ["Failure Rate", statistics && statistics.trace_count > 0 ? `${Math.round((failureCount / statistics.trace_count) * 100)}%` : "—"],
          ["Avg Confidence", statistics ? `${Math.round(statistics.average_confidence * 100)}%` : "—"],
          ["Avg Latency", avgLatency !== null ? `${avgLatency.toFixed(3)}s` : "—"],
          ["Cache Hit Ratio", cacheHitRatio !== null ? `${Math.round(cacheHitRatio * 100)}%` : "—"],
          ["Optimization Suggestions", recommendations.length],
          ["Pattern Count", patterns.length],
          ["Most Frequent Failure", mostFrequent?.category ?? "none"],
        ].map(([label, value]) => (
          <div key={label as string} className="rounded border border-slate-200 p-3">
            <p className="text-xs text-slate-500">{label}</p>
            <p className="text-lg font-semibold text-slate-900">{value}</p>
          </div>
        ))}
      </div>
      {trendData.length > 0 && (
        <div className="mt-4" style={{ width: "100%", height: 180 }}>
          <ResponsiveContainer>
            <BarChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" tick={{ fontSize: 9 }} interval={0} angle={-20} textAnchor="end" height={60} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Bar dataKey="support" fill="#4f46e5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function FailurePatternExplorer() {
  const [patterns, setPatterns] = useState<FailurePattern[]>([]);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState("");
  const [severityFilter, setSeverityFilter] = useState("");
  const [selectedPatternId, setSelectedPatternId] = useState<string | null>(null);
  const [attribution, setAttribution] = useState<FailureAttribution | null>(null);

  async function refresh() {
    setLoading("refresh");
    setError(null);
    try {
      setPatterns(await listPatterns());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load patterns");
    } finally {
      setLoading(null);
    }
  }

  async function handleMine() {
    setLoading("mine");
    setError(null);
    try {
      setPatterns(await minePatterns());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to mine patterns");
    } finally {
      setLoading(null);
    }
  }

  async function handleSelectPattern(patternId: string) {
    setSelectedPatternId(patternId);
    setAttribution(null);
    try {
      setAttribution(await getPatternAttribution(patternId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load attribution");
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const categories = useMemo(() => Array.from(new Set(patterns.map((p) => p.category))).sort(), [patterns]);
  const filtered = patterns.filter((p) => (!categoryFilter || p.category === categoryFilter) && (!severityFilter || p.severity === severityFilter));
  const selectedPattern = patterns.find((p) => p.pattern_id === selectedPatternId) ?? null;

  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Failure Pattern Explorer</h2>
        <button onClick={handleMine} disabled={loading === "mine"} className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
          {loading === "mine" ? "Mining..." : "Mine Patterns"}
        </button>
      </div>
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      <div className="mb-3 flex gap-3">
        <select value={categoryFilter} onChange={(e) => setCategoryFilter(e.target.value)} className="rounded border border-slate-300 px-2 py-1 text-sm">
          <option value="">All categories</option>
          {categories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)} className="rounded border border-slate-300 px-2 py-1 text-sm">
          <option value="">All severities</option>
          {["low", "medium", "high", "critical"].map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
      </div>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="text-slate-500">
            <th className="pb-1">Category</th>
            <th className="pb-1">Support</th>
            <th className="pb-1">Frequency</th>
            <th className="pb-1">Severity</th>
            <th className="pb-1" />
          </tr>
        </thead>
        <tbody>
          {filtered.map((p) => (
            <tr key={p.pattern_id} className="border-t border-slate-100">
              <td className="py-1">{p.category}</td>
              <td className="py-1">{p.support}</td>
              <td className="py-1">{Math.round(p.frequency * 100)}%</td>
              <td className="py-1">
                <span className={`rounded px-1.5 py-0.5 text-xs ${severityBadgeClass(p.severity)}`}>{p.severity}</span>
              </td>
              <td className="py-1 text-right">
                <button onClick={() => handleSelectPattern(p.pattern_id)} className="rounded border border-slate-300 px-2 py-0.5 text-xs">
                  Details
                </button>
              </td>
            </tr>
          ))}
          {filtered.length === 0 && (
            <tr>
              <td colSpan={5} className="py-3 text-center text-slate-400">
                No patterns match the current filters.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {selectedPattern && (
        <div className="mt-4 rounded border border-slate-200 p-3">
          <p className="text-sm text-slate-700">{selectedPattern.description}</p>
          {attribution && (
            <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-slate-600">
              <dt>Primary cause</dt>
              <dd>{attribution.primary_cause}</dd>
              <dt>Secondary cause</dt>
              <dd>{attribution.secondary_cause ?? "—"}</dd>
              <dt>Evidence confidence</dt>
              <dd>{Math.round(attribution.evidence_confidence * 100)}%</dd>
              <dt>Execution confidence</dt>
              <dd>{Math.round(attribution.execution_confidence * 100)}%</dd>
            </dl>
          )}
          <h4 className="mt-3 text-xs font-medium text-slate-700">Supporting executions</h4>
          <div className="mt-1 flex flex-col gap-1">
            {selectedPattern.example_execution_ids.map((id) => (
              <ExecutionSummaryCard key={id} executionId={id} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function RootCauseGraphVisualization() {
  const [executionId, setExecutionId] = useState("");
  const [graph, setGraph] = useState<RootCauseGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [pathStart, setPathStart] = useState<string | null>(null);

  async function handleLoad() {
    if (!executionId.trim()) return;
    setLoading(true);
    setError(null);
    setGraph(null);
    setSelectedNodeId(null);
    setPathStart(null);
    try {
      setGraph(await getRootCauseGraph(executionId.trim()));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load root cause graph");
    } finally {
      setLoading(false);
    }
  }

  function handleNodeTap(nodeId: string) {
    if (pathStart && pathStart !== nodeId) {
      setSelectedNodeId(nodeId);
      return;
    }
    setSelectedNodeId(nodeId);
  }

  function handleExport() {
    if (!graph) return;
    const blob = new Blob([JSON.stringify(graph, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `root-cause-graph-${executionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const elements = useMemo<ElementDefinition[]>(() => {
    if (!graph) return [];
    const nodeElements: ElementDefinition[] = graph.nodes.map((n) => ({ data: { id: n.id, label: n.label, node_type: n.type } }));
    const edgeElements: ElementDefinition[] = graph.edges.map((e, i) => ({ data: { id: `rcg-edge-${i}`, source: e.source, target: e.target, label: e.type } }));
    return [...nodeElements, ...edgeElements];
  }, [graph]);

  const stylesheet = useMemo<Stylesheet[]>(
    () => [
      {
        selector: "node",
        style: {
          "background-color": (ele: NodeSingular) => NODE_TYPE_COLORS[ele.data("node_type") as string] ?? "#94a3b8",
          label: "data(label)",
          "font-size": 8,
          width: 26,
          height: 26,
          "text-valign": "bottom",
          "text-margin-y": 4,
        },
      },
      { selector: "edge", style: { width: 1.5, "line-color": "#cbd5e1", "target-arrow-color": "#cbd5e1", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", "font-size": 6 } },
      { selector: "node.selected", style: { "border-width": 3, "border-color": "#4f46e5" } },
      { selector: "node.highlighted", style: { "border-width": 3, "border-color": "#f59e0b" } },
    ],
    [],
  );

  const selectedNode = graph?.nodes.find((n) => n.id === selectedNodeId) ?? null;
  const dominantCause = selectedNode && selectedNode.type === "failure" && graph ? findDominantCause(graph, selectedNode.id) : null;
  const dependencyDepth = selectedNode && graph ? computeDependencyDepth(graph, selectedNode.id) : null;
  const criticalPath = pathStart && selectedNodeId && graph ? findCriticalPath(graph.edges, pathStart, selectedNodeId) : null;
  const highlightedIds = useMemo(() => (criticalPath ? new Set(criticalPath) : null), [criticalPath]);

  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Root Cause Graph</h2>
      <div className="mb-3 flex flex-wrap items-end gap-2">
        <label className="text-sm text-slate-600">
          Execution id
          <input
            type="text"
            value={executionId}
            onChange={(e) => setExecutionId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            className="mt-1 block w-64 rounded border border-slate-300 px-3 py-1.5"
          />
        </label>
        <button onClick={handleLoad} disabled={loading} className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
          {loading ? "Loading..." : "Load Graph"}
        </button>
        {graph && (
          <button onClick={handleExport} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50">
            Export JSON
          </button>
        )}
        {selectedNodeId && (
          <button onClick={() => setPathStart(selectedNodeId)} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
            Set as path start
          </button>
        )}
      </div>
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      {graph && (
        <>
          <CytoscapeGraph elements={elements} stylesheet={stylesheet} onNodeTap={handleNodeTap} selectedNodeId={selectedNodeId} highlightedNodeIds={highlightedIds} testId="root-cause-graph-canvas" />
          <div className="mt-3 grid grid-cols-1 gap-2 text-xs text-slate-600 sm:grid-cols-3">
            <p>
              <span className="font-medium">Selected node:</span> {selectedNode ? `${selectedNode.label} (${selectedNode.type})` : "none"}
            </p>
            <p>
              <span className="font-medium">Dominant cause:</span> {dominantCause ? `${dominantCause.label} (${dominantCause.type})` : "n/a"}
            </p>
            <p>
              <span className="font-medium">Dependency depth:</span> {dependencyDepth ?? "n/a"}
            </p>
          </div>
          {pathStart && (
            <p className="mt-2 text-xs text-slate-600">
              <span className="font-medium">Critical path from {pathStart}:</span> {criticalPath ? criticalPath.join(" → ") : "no path found"}
            </p>
          )}
        </>
      )}
    </div>
  );
}

function CounterfactualExplorer() {
  const [executionId, setExecutionId] = useState("");
  const [toggle, setToggle] = useState<string>(COUNTERFACTUAL_TOGGLES[0]);
  const [result, setResult] = useState<CounterfactualResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleRun() {
    if (!executionId.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await simulateCounterfactual(executionId.trim(), toggle));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Counterfactual simulation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Counterfactual Explorer</h2>
      <div className="flex flex-wrap items-end gap-2">
        <label className="text-sm text-slate-600">
          Execution id
          <input type="text" value={executionId} onChange={(e) => setExecutionId(e.target.value)} className="mt-1 block w-64 rounded border border-slate-300 px-3 py-1.5" />
        </label>
        <label className="text-sm text-slate-600">
          Toggle
          <select value={toggle} onChange={(e) => setToggle(e.target.value)} className="mt-1 block rounded border border-slate-300 px-3 py-1.5">
            {COUNTERFACTUAL_TOGGLES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </label>
        <button onClick={handleRun} disabled={loading} className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">
          {loading ? "Simulating..." : "Run Simulation"}
        </button>
      </div>
      {error && <p className="mt-2 rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      {result && (
        <div className="mt-3 rounded border border-slate-200 p-3 text-sm">
          <p>
            <span className="font-medium">Applicable:</span> {result.applicable ? "yes" : "no"}
          </p>
          <p>
            <span className="font-medium">Baseline failures:</span> {result.baseline_failure_categories.join(", ") || "none"}
          </p>
          <p>
            <span className="font-medium">Simulated failures:</span> {result.simulated_failure_categories.join(", ") || "none"}
          </p>
          <p>
            <span className="font-medium">Disappeared:</span> {result.disappeared_categories.join(", ") || "none"}
          </p>
          <p>
            <span className="font-medium">Expected improvement:</span> {Math.round(result.expected_improvement * 100)}%
          </p>
          <p>
            <span className="font-medium">Confidence:</span> {Math.round(result.confidence * 100)}%
          </p>
          <p className="mt-1 text-xs text-slate-500">{result.note}</p>
        </div>
      )}
    </div>
  );
}

function OptimizationRecommendationViewer() {
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  useEffect(() => {
    getOptimizationRecommendations()
      .then(setRecommendations)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load recommendations"));
  }, []);

  return (
    <div className="rounded-lg border border-slate-200 p-4">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">Optimization Recommendations</h2>
      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      <ul className="flex flex-col gap-2">
        {recommendations.map((rec, i) => (
          <li key={i} className="rounded border border-slate-200 p-3 text-sm">
            <div className="flex items-center justify-between">
              <p className="font-medium text-slate-800">{rec.subsystem}</p>
              <span className="rounded bg-indigo-50 px-1.5 py-0.5 text-xs text-indigo-700">{Math.round(rec.confidence * 100)}% confidence</span>
            </div>
            <p className="mt-1 text-slate-600">{rec.explanation}</p>
            <p className="mt-1 text-xs text-slate-500">{rec.expected_benefit}</p>
            <button onClick={() => setExpandedIndex(expandedIndex === i ? null : i)} className="mt-2 text-xs font-medium text-indigo-600">
              {expandedIndex === i ? "Hide supporting evidence" : "Show supporting evidence"}
            </button>
            {expandedIndex === i && (
              <div className="mt-2 flex flex-col gap-2">
                <ul className="list-inside list-disc text-xs text-slate-600">
                  {rec.evidence.map((e, ei) => (
                    <li key={ei}>{e}</li>
                  ))}
                </ul>
                {rec.supporting_execution_ids.map((id) => (
                  <ExecutionSummaryCard key={id} executionId={id} />
                ))}
              </div>
            )}
          </li>
        ))}
        {recommendations.length === 0 && !error && <p className="text-sm text-slate-400">No optimization recommendations yet — mine some failure patterns first.</p>}
      </ul>
    </div>
  );
}

export function FailureIntelligencePage() {
  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      <h1 className="text-xl font-semibold text-slate-900">Failure Intelligence</h1>
      <DashboardMetrics />
      <FailurePatternExplorer />
      <RootCauseGraphVisualization />
      <CounterfactualExplorer />
      <OptimizationRecommendationViewer />
    </div>
  );
}
