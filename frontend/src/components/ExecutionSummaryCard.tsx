import { useEffect, useState } from "react";
import { getExecution } from "../services/api";
import type { ExecutionTrace } from "../services/api";

/**
 * Small drill-down card: fetches and displays one execution's key fields
 * given its id. Shared by the Failure Pattern Explorer and Optimization
 * Recommendation Viewer (both need to preview "supporting executions")
 * rather than each re-fetching/re-rendering the same summary inline.
 */
export function ExecutionSummaryCard({ executionId }: { executionId: string }) {
  const [trace, setTrace] = useState<ExecutionTrace | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTrace(null);
    setError(null);
    getExecution(executionId)
      .then(setTrace)
      .catch((err) => setError(err instanceof Error ? err.message : "Failed to load execution"));
  }, [executionId]);

  if (error) {
    return <p className="rounded border border-red-200 bg-red-50 p-2 text-xs text-red-700">{error}</p>;
  }
  if (!trace) {
    return <p className="rounded border border-slate-200 p-2 text-xs text-slate-400">Loading {executionId}…</p>;
  }
  return (
    <div className="rounded border border-slate-200 p-2 text-xs">
      <p className="font-medium text-slate-800">{trace.query}</p>
      <p className="text-slate-500">
        confidence {Math.round(trace.confidence * 100)}% · {trace.events.length} event(s) · {trace.timestamps.started_at}
      </p>
    </div>
  );
}
