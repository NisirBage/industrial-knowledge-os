import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AUTO_RUN_BENCHMARKS,
  COMPRESSION_LEVELS,
  OPTIMIZATION_BENCHMARKS,
  getBenchmarkHistory,
  getCalibration,
  getEntityDuplicates,
  listExperiments,
  mergeEntities,
  previewContextCompression,
  runBenchmark,
  runEntityExtractionABTest,
} from "../services/api";
import type {
  ABTestResult,
  BenchmarkReport,
  CalibrationBucket,
  CompressionPreview,
  DuplicateCandidate,
  Experiment,
} from "../services/api";

export function ResearchPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>(AUTO_RUN_BENCHMARKS[0]);
  const [history, setHistory] = useState<Experiment[]>([]);
  const [lastReport, setLastReport] = useState<BenchmarkReport | null>(null);
  const [abResult, setAbResult] = useState<ABTestResult | null>(null);
  const [calibration, setCalibration] = useState<CalibrationBucket[]>([]);
  const [optimizationResults, setOptimizationResults] = useState<Record<string, BenchmarkReport>>({});
  const [compressionQuery, setCompressionQuery] = useState("");
  const [compressionLevel, setCompressionLevel] = useState<string>(COMPRESSION_LEVELS[1]);
  const [compressionPreview, setCompressionPreview] = useState<CompressionPreview | null>(null);
  const [duplicates, setDuplicates] = useState<DuplicateCandidate[]>([]);
  const [mergedIds, setMergedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refreshExperiments() {
    setExperiments(await listExperiments());
  }

  async function refreshHistory(benchmarkName: string) {
    setHistory(await getBenchmarkHistory(benchmarkName));
  }

  async function refreshCalibration() {
    const result = await getCalibration();
    setCalibration(result.buckets);
  }

  useEffect(() => {
    refreshExperiments().catch((err) => setError(String(err)));
    refreshCalibration().catch(() => undefined);
  }, []);

  useEffect(() => {
    refreshHistory(selectedBenchmark).catch(() => undefined);
  }, [selectedBenchmark]);

  async function handleRunBenchmark() {
    setLoading("benchmark");
    setError(null);
    try {
      const report = await runBenchmark(selectedBenchmark);
      setLastReport(report);
      await Promise.all([refreshExperiments(), refreshHistory(selectedBenchmark)]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark run failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleRunOptimizationBenchmark(name: string) {
    setLoading(`optimization:${name}`);
    setError(null);
    try {
      const report = await runBenchmark(name);
      setOptimizationResults((prev) => ({ ...prev, [name]: report }));
    } catch (err) {
      setError(err instanceof Error ? err.message : `Benchmark ${name} failed`);
    } finally {
      setLoading(null);
    }
  }

  async function handleRunABTest() {
    setLoading("abtest");
    setError(null);
    try {
      setAbResult(await runEntityExtractionABTest());
    } catch (err) {
      setError(err instanceof Error ? err.message : "A/B test failed");
    } finally {
      setLoading(null);
    }
  }

  async function handlePreviewCompression() {
    if (!compressionQuery.trim()) return;
    setLoading("compression");
    setError(null);
    try {
      setCompressionPreview(await previewContextCompression(compressionQuery, compressionLevel));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Compression preview failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleFindDuplicates() {
    setLoading("duplicates");
    setError(null);
    try {
      setDuplicates(await getEntityDuplicates());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Duplicate lookup failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleMerge(sourceId: string, targetId: string) {
    setLoading(`merge:${sourceId}`);
    setError(null);
    try {
      await mergeEntities(sourceId, targetId);
      setMergedIds((prev) => new Set(prev).add(sourceId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Merge failed");
    } finally {
      setLoading(null);
    }
  }

  const latencyTrend = history
    .slice()
    .reverse()
    .map((e, i) => ({ run: i + 1, duration: e.duration ?? 0 }));

  const f1Trend = history
    .slice()
    .reverse()
    .map((e, i) => ({ run: i + 1, f1: e.metrics.f1 ?? 0 }));

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-6">
      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Benchmark Suite</h2>
        <div className="flex items-center gap-2">
          <select
            value={selectedBenchmark}
            onChange={(e) => setSelectedBenchmark(e.target.value)}
            className="rounded border border-slate-300 px-3 py-1.5"
          >
            {AUTO_RUN_BENCHMARKS.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <button
            onClick={handleRunBenchmark}
            disabled={loading === "benchmark"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 font-medium text-white disabled:opacity-50"
          >
            {loading === "benchmark" ? "Running..." : "Run Benchmark"}
          </button>
        </div>

        {lastReport && (
          <div className="mt-3 text-sm text-slate-700">
            <p className="font-medium">Latest result ({lastReport.duration_seconds.toFixed(3)}s, n={lastReport.sample_size}):</p>
            <ul className="list-inside list-disc">
              {lastReport.metrics.map((m) => (
                <li key={m.name}>
                  {m.name}: {m.value}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Evaluation Trend (F1) — {selectedBenchmark}</h2>
        {f1Trend.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={f1Trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="run" label={{ value: "Run #", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Line type="monotone" dataKey="f1" stroke="#4f46e5" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-400">No runs yet — run a benchmark above.</p>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Latency Trend — {selectedBenchmark}</h2>
        {latencyTrend.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={latencyTrend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="run" label={{ value: "Run #", position: "insideBottom", offset: -2 }} />
              <YAxis label={{ value: "seconds", angle: -90, position: "insideLeft" }} />
              <Tooltip />
              <Line type="monotone" dataKey="duration" stroke="#059669" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-400">No runs yet.</p>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">A/B Test: Regex-only vs Production Extraction</h2>
          <button
            onClick={handleRunABTest}
            disabled={loading === "abtest"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading === "abtest" ? "Running..." : "Run A/B Test"}
          </button>
        </div>
        {abResult && (
          <div className="text-sm text-slate-700">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={[{ name: abResult.metric_name, [abResult.arm_a_name]: abResult.mean_a, [abResult.arm_b_name]: abResult.mean_b }]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar dataKey={abResult.arm_a_name} fill="#94a3b8" />
                <Bar dataKey={abResult.arm_b_name} fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-2">
              Winner: <span className="font-medium">{abResult.winner}</span> — delta {abResult.mean_delta}, p-value {abResult.p_value}
            </p>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Confidence Calibration</h2>
        {calibration.length > 0 ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={calibration}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="confidence_bucket" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="mean_confidence" fill="#94a3b8" name="Mean Confidence" />
              <Bar dataKey="actual_accuracy" fill="#4f46e5" name="Actual Accuracy" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-sm text-slate-400">No predictions with recorded outcomes yet.</p>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Context Compression Preview</h2>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={compressionQuery}
            onChange={(e) => setCompressionQuery(e.target.value)}
            placeholder="e.g. pump P-101 maintenance and failures"
            className="min-w-[240px] flex-1 rounded border border-slate-300 px-3 py-1.5"
          />
          <select
            value={compressionLevel}
            onChange={(e) => setCompressionLevel(e.target.value)}
            className="rounded border border-slate-300 px-3 py-1.5"
          >
            {COMPRESSION_LEVELS.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
          <button
            onClick={handlePreviewCompression}
            disabled={loading === "compression"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading === "compression" ? "Running..." : "Preview Compression"}
          </button>
        </div>
        {compressionPreview && (
          <div className="mt-3 text-sm text-slate-700">
            <p className="font-medium">{compressionPreview.stats.summary}</p>
            <p className="mt-1 text-slate-500">
              compression_ratio={compressionPreview.stats.compression_ratio}, token_reduction=
              {(compressionPreview.stats.token_reduction_ratio * 100).toFixed(1)}%
            </p>
          </div>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Entity Resolution — Duplicate Candidates</h2>
          <button
            onClick={handleFindDuplicates}
            disabled={loading === "duplicates"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading === "duplicates" ? "Running..." : "Find Duplicates"}
          </button>
        </div>
        {duplicates.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Source</th>
                <th className="pb-1">Target</th>
                <th className="pb-1">Confidence</th>
                <th className="pb-1">Reason</th>
                <th className="pb-1"></th>
              </tr>
            </thead>
            <tbody>
              {duplicates.map((d) => (
                <tr key={`${d.source_id}-${d.target_id}`} className="border-t border-slate-100">
                  <td className="py-1">{d.source_name}</td>
                  <td className="py-1">{d.target_name}</td>
                  <td className="py-1">{d.confidence}</td>
                  <td className="py-1 text-slate-500">{d.merge_reason}</td>
                  <td className="py-1">
                    {mergedIds.has(d.source_id) ? (
                      <span className="text-emerald-600">Merged</span>
                    ) : (
                      <button
                        onClick={() => handleMerge(d.source_id, d.target_id)}
                        disabled={loading === `merge:${d.source_id}`}
                        className="rounded border border-slate-300 px-2 py-0.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                      >
                        {loading === `merge:${d.source_id}` ? "Merging..." : "Merge"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Optimization Benchmarks (Learning-to-Rank, Compression, Entity Resolution)</h2>
        <div className="flex flex-wrap gap-2">
          {OPTIMIZATION_BENCHMARKS.map((name) => (
            <button
              key={name}
              onClick={() => handleRunOptimizationBenchmark(name)}
              disabled={loading === `optimization:${name}`}
              className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              {loading === `optimization:${name}` ? "Running..." : name}
            </button>
          ))}
        </div>
        {Object.keys(optimizationResults).length > 0 && (
          <table className="mt-3 w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Benchmark</th>
                <th className="pb-1">Metrics</th>
                <th className="pb-1">Latency</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(optimizationResults).map(([name, report]) => (
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

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Experiment History</h2>
        {experiments.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Name</th>
                <th className="pb-1">Status</th>
                <th className="pb-1">Dataset</th>
                <th className="pb-1">Duration</th>
                <th className="pb-1">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {experiments.slice(0, 15).map((e) => (
                <tr key={e.experiment_id} className="border-t border-slate-100">
                  <td className="py-1">{e.name}</td>
                  <td className="py-1">{e.status}</td>
                  <td className="py-1">{e.dataset}</td>
                  <td className="py-1">{e.duration?.toFixed(3) ?? "-"}s</td>
                  <td className="py-1 text-xs text-slate-500">{e.timestamp}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-slate-400">No experiments recorded yet.</p>
        )}
      </div>
    </div>
  );
}
