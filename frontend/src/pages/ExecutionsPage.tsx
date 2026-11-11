import type { ElementDefinition, StylesheetJsonBlock as Stylesheet } from "cytoscape";
import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CytoscapeGraph } from "../components/CytoscapeGraph";
import {
  getExecution,
  getExecutionTimeline,
  listExecutions,
  listPatterns,
} from "../services/api";
import type { ExecutionTrace, FailurePattern, ReconstructedTimeline } from "../services/api";

const PAGE_SIZE = 10;

type SortKey = "started_at" | "confidence" | "event_count";

function confidenceBadgeClass(confidence: number): string {
  if (confidence >= 0.7) return "bg-emerald-50 text-emerald-700";
  if (confidence >= 0.4) return "bg-amber-50 text-amber-700";
  return "bg-red-50 text-red-700";
}

function formatSeconds(seconds: number | null): string {
  return seconds === null ? "—" : `${seconds.toFixed(3)}s`;
}

function ExecutionInspector({ executionId }: { executionId: string }) {
  const [trace, setTrace] = useState<ExecutionTrace | null>(null);
  const [timeline, setTimeline] = useState<ReconstructedTimeline | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedComponent, setExpandedComponent] = useState<string | null>(null);

  useEffect(() => {
    setTrace(null);
    setTimeline(null);
    setError(null);
    Promise.all([getExecution(executionId), getExecutionTimeline(executionId)])
      .then(([nextTrace, nextTimeline]) => {
        setTrace(nextTrace);
        setTimeline(nextTimeline);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load execution"));
  }, [executionId]);

  const waterfallData = useMemo(() => {
    if (!timeline || timeline.spans.length === 0) return [];
    const t0 = new Date(timeline.spans[0].started_at).getTime();
    return timeline.spans.map((s) => ({
      component: s.component,
      offset: Math.max(0, (new Date(s.started_at).getTime() - t0) / 1000),
      duration: s.duration_seconds,
      confidence: s.confidence,
      memory_bytes: s.memory_bytes,
    }));
  }, [timeline]);

  const confidenceEvolutionData = useMemo(
    () => (timeline?.confidence_evolution ?? []).map(([component, confidence], i) => ({ index: i, component, confidence })),
    [timeline],
  );

  const dagStylesheet = useMemo<Stylesheet[]>(
    () => [
      { selector: "node", style: { "background-color": "#4f46e5", label: "data(label)", "font-size": 8, width: 24, height: 24, "text-valign": "bottom", "text-margin-y": 4 } },
      { selector: "edge", style: { width: 1.5, "line-color": "#cbd5e1", "target-arrow-color": "#cbd5e1", "target-arrow-shape": "triangle", "curve-style": "bezier" } },
    ],
    [],
  );

  const dagElements = useMemo<ElementDefinition[]>(() => {
    if (!timeline) return [];
    const nodeElements: ElementDefinition[] = timeline.dag.nodes.map((n) => ({ data: { id: n, label: n } }));
    const edgeElements: ElementDefinition[] = timeline.dag.edges.map(([source, target], i) => ({
      data: { id: `dag-edge-${i}`, source, target },
    }));
    return [...nodeElements, ...edgeElements];
  }, [timeline]);

  if (error) return <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>;
  if (!trace || !timeline) return <p className="text-sm text-slate-500">Loading execution…</p>;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className={`rounded px-2 py-0.5 ${confidenceBadgeClass(trace.confidence)}`}>
          confidence {Math.round(trace.confidence * 100)}%
        </span>
        <span className="text-slate-500">{trace.events.length} event(s)</span>
        <span className="text-slate-500">started {trace.timestamps.started_at}</span>
      </div>
      <p className="text-sm text-slate-700">
        <span className="font-medium">Query:</span> {trace.query}
      </p>

      {/* Pipeline Waterfall */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-800">Pipeline Waterfall</h3>
        {waterfallData.length === 0 ? (
          <p className="text-sm text-slate-400">No stage timing recorded.</p>
        ) : (
          <div style={{ width: "100%", height: Math.max(120, waterfallData.length * 32) }}>
            <ResponsiveContainer>
              <BarChart data={waterfallData} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={(v: number) => `${v}s`} />
                <YAxis type="category" dataKey="component" width={120} tick={{ fontSize: 11 }} />
                <Tooltip formatter={(value, name) => [`${Number(value).toFixed(3)}s`, String(name)]} />
                <Bar dataKey="offset" stackId="a" fill="transparent" isAnimationActive={false} />
                <Bar dataKey="duration" stackId="a" fill="#4f46e5" isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
        <ul className="mt-3 flex flex-col gap-1">
          {timeline.spans.map((span) => (
            <li key={span.component} className="rounded border border-slate-200 text-xs">
              <button
                onClick={() => setExpandedComponent(expandedComponent === span.component ? null : span.component)}
                className="flex w-full items-center justify-between px-2 py-1 text-left"
              >
                <span className="font-medium text-slate-800">{span.component}</span>
                <span className="text-slate-500">
                  {formatSeconds(span.duration_seconds)}
                  {span.confidence !== null && ` · confidence ${Math.round(span.confidence * 100)}%`}
                  {timeline.cache_hits.includes(span.component) && " · cache hit"}
                </span>
              </button>
              {expandedComponent === span.component && (
                <dl className="grid grid-cols-2 gap-x-2 gap-y-0.5 border-t border-slate-100 p-2 text-[11px] text-slate-600">
                  <dt>Started</dt>
                  <dd>{span.started_at}</dd>
                  <dt>Finished</dt>
                  <dd>{span.finished_at}</dd>
                  <dt>Memory</dt>
                  <dd>{span.memory_bytes ?? "—"}</dd>
                </dl>
              )}
            </li>
          ))}
        </ul>
      </div>

      {/* Confidence Evolution */}
      {confidenceEvolutionData.length > 0 && (
        <div>
          <h3 className="mb-2 text-sm font-semibold text-slate-800">Confidence Evolution</h3>
          <div style={{ width: "100%", height: 180 }}>
            <ResponsiveContainer>
              <LineChart data={confidenceEvolutionData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="component" tick={{ fontSize: 10 }} />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Line type="monotone" dataKey="confidence" stroke="#059669" strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Execution DAG */}
      <div>
        <h3 className="mb-2 text-sm font-semibold text-slate-800">Execution DAG</h3>
        {dagElements.length === 0 ? (
          <p className="text-sm text-slate-400">No DAG data available.</p>
        ) : (
          <CytoscapeGraph elements={dagElements} stylesheet={dagStylesheet} height={280} testId="execution-dag-canvas" />
        )}
      </div>
    </div>
  );
}

export function ExecutionsPage() {
  const [executions, setExecutions] = useState<ExecutionTrace[]>([]);
  const [patterns, setPatterns] = useState<FailurePattern[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [componentFilter, setComponentFilter] = useState("");
  const [failureOnly, setFailureOnly] = useState(false);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("started_at");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(1);

  const [lookupId, setLookupId] = useState("");
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const [nextExecutions, nextPatterns] = await Promise.all([listExecutions(100), listPatterns()]);
      setExecutions(nextExecutions);
      setPatterns(nextPatterns);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load executions");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh().catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const failureExecutionIds = useMemo(() => new Set(patterns.flatMap((p) => p.example_execution_ids)), [patterns]);

  const componentOptions = useMemo(() => {
    const set = new Set<string>();
    executions.forEach((e) => e.events.forEach((ev) => set.add(ev.component)));
    return Array.from(set).sort();
  }, [executions]);

  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return executions.filter((e) => {
      if (needle && !e.query.toLowerCase().includes(needle) && !e.execution_id.toLowerCase().includes(needle)) return false;
      if (componentFilter && !e.events.some((ev) => ev.component === componentFilter)) return false;
      if (failureOnly && !failureExecutionIds.has(e.execution_id)) return false;
      if (dateFrom && e.timestamps.started_at < dateFrom) return false;
      if (dateTo && e.timestamps.started_at > `${dateTo}T23:59:59`) return false;
      return true;
    });
  }, [executions, search, componentFilter, failureOnly, failureExecutionIds, dateFrom, dateTo]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    copy.sort((a, b) => {
      let cmp: number;
      if (sortKey === "confidence") cmp = a.confidence - b.confidence;
      else if (sortKey === "event_count") cmp = a.events.length - b.events.length;
      else cmp = a.timestamps.started_at.localeCompare(b.timestamps.started_at);
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [filtered, sortKey, sortDir]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const paged = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  function toggleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
    setPage(1);
  }

  async function handleLookup() {
    if (!lookupId.trim()) return;
    setError(null);
    try {
      await getExecution(lookupId.trim());
      setSelectedExecutionId(lookupId.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Execution not found");
    }
  }

  return (
    <div className="mx-auto flex max-w-7xl flex-col gap-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-slate-900">Execution Explorer</h1>
        <button onClick={() => refresh()} disabled={loading} className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50">
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      <div className="flex flex-wrap items-end gap-3 rounded-lg border border-slate-200 p-4">
        <label className="text-sm text-slate-600">
          Search (query or execution id)
          <input
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="mt-1 block w-64 rounded border border-slate-300 px-3 py-1.5"
          />
        </label>
        <label className="text-sm text-slate-600">
          Component
          <select
            value={componentFilter}
            onChange={(e) => {
              setComponentFilter(e.target.value);
              setPage(1);
            }}
            className="mt-1 block rounded border border-slate-300 px-3 py-1.5"
          >
            <option value="">All</option>
            {componentOptions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        <label className="text-sm text-slate-600">
          From
          <input type="date" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} className="mt-1 block rounded border border-slate-300 px-3 py-1.5" />
        </label>
        <label className="text-sm text-slate-600">
          To
          <input type="date" value={dateTo} onChange={(e) => setDateTo(e.target.value)} className="mt-1 block rounded border border-slate-300 px-3 py-1.5" />
        </label>
        <label className="flex items-center gap-2 text-sm text-slate-600">
          <input type="checkbox" checked={failureOnly} onChange={(e) => setFailureOnly(e.target.checked)} />
          Only executions with detected failures
        </label>
        <div className="ml-auto flex items-end gap-2">
          <label className="text-sm text-slate-600">
            Look up execution id
            <input
              type="text"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLookup()}
              className="mt-1 block w-56 rounded border border-slate-300 px-3 py-1.5"
            />
          </label>
          <button onClick={handleLookup} className="rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-medium text-white">
            Open
          </button>
        </div>
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="text-slate-500">
              <th className="cursor-pointer pb-2" onClick={() => toggleSort("started_at")}>
                Started {sortKey === "started_at" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="pb-2">Query</th>
              <th className="cursor-pointer pb-2" onClick={() => toggleSort("confidence")}>
                Confidence {sortKey === "confidence" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="cursor-pointer pb-2" onClick={() => toggleSort("event_count")}>
                Events {sortKey === "event_count" && (sortDir === "asc" ? "↑" : "↓")}
              </th>
              <th className="pb-2" />
            </tr>
          </thead>
          <tbody>
            {paged.map((execution) => (
              <tr key={execution.execution_id} className="border-t border-slate-100">
                <td className="py-1.5 text-xs text-slate-500">{execution.timestamps.started_at}</td>
                <td className="py-1.5 max-w-xs truncate" title={execution.query}>
                  {execution.query}
                </td>
                <td className="py-1.5">
                  <span className={`rounded px-1.5 py-0.5 text-xs ${confidenceBadgeClass(execution.confidence)}`}>{Math.round(execution.confidence * 100)}%</span>
                </td>
                <td className="py-1.5 text-xs text-slate-500">{execution.events.length}</td>
                <td className="py-1.5 text-right">
                  <button onClick={() => setSelectedExecutionId(execution.execution_id)} className="rounded border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-700 hover:bg-slate-50">
                    View
                  </button>
                </td>
              </tr>
            ))}
            {paged.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center text-sm text-slate-400">
                  No executions match the current filters.
                </td>
              </tr>
            )}
          </tbody>
        </table>
        {pageCount > 1 && (
          <div className="mt-3 flex items-center justify-between text-sm">
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)} className="rounded border border-slate-300 px-2 py-1 disabled:opacity-50">
              Previous
            </button>
            <span className="text-slate-500">
              Page {page} of {pageCount}
            </span>
            <button disabled={page >= pageCount} onClick={() => setPage((p) => p + 1)} className="rounded border border-slate-300 px-2 py-1 disabled:opacity-50">
              Next
            </button>
          </div>
        )}
      </div>

      {selectedExecutionId && (
        <div className="rounded-lg border border-slate-200 p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900">Execution Inspector — {selectedExecutionId}</h2>
            <button onClick={() => setSelectedExecutionId(null)} className="rounded border border-slate-300 px-2 py-0.5 text-xs">
              Close
            </button>
          </div>
          <ExecutionInspector executionId={selectedExecutionId} />
        </div>
      )}
    </div>
  );
}
