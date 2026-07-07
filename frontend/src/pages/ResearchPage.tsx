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
  getBenchmarkHistory,
  getCalibration,
  listExperiments,
  runBenchmark,
  runEntityExtractionABTest,
} from "../services/api";
import type { ABTestResult, BenchmarkReport, CalibrationBucket, Experiment } from "../services/api";

export function ResearchPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string>(AUTO_RUN_BENCHMARKS[0]);
  const [history, setHistory] = useState<Experiment[]>([]);
  const [lastReport, setLastReport] = useState<BenchmarkReport | null>(null);
  const [abResult, setAbResult] = useState<ABTestResult | null>(null);
  const [calibration, setCalibration] = useState<CalibrationBucket[]>([]);
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
