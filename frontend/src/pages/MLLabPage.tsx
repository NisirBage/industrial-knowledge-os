import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  TRAINABLE_ML_TASKS,
  computeDrift,
  getFeatureDefinitions,
  listModels,
  trainModel,
} from "../services/api";
import type { DriftResult, FeatureDefinition, ModelRecord, MLTrainingResult } from "../services/api";

export function MLLabPage() {
  const [taskType, setTaskType] = useState<string>(TRAINABLE_ML_TASKS[0]);
  const [sampleSize, setSampleSize] = useState(300);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<MLTrainingResult | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [featureDefs, setFeatureDefs] = useState<FeatureDefinition[]>([]);
  const [drift, setDrift] = useState<DriftResult | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function refreshModels() {
    setModels(await listModels());
  }

  useEffect(() => {
    refreshModels().catch((err) => setError(String(err)));
    getFeatureDefinitions()
      .then(setFeatureDefs)
      .catch(() => undefined);
  }, []);

  async function handleTrain() {
    setLoading("train");
    setError(null);
    try {
      const trained = await trainModel(taskType, sampleSize, seed);
      setResult(trained);
      await refreshModels();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Training failed");
    } finally {
      setLoading(null);
    }
  }

  async function handleDriftDemo() {
    setLoading("drift");
    setError(null);
    try {
      const baseline = Array.from({ length: 50 }, (_, i) => 0.2 + (i % 5) * 0.05);
      const recent = Array.from({ length: 50 }, (_, i) => 0.5 + (i % 5) * 0.05);
      setDrift(await computeDrift(baseline, recent));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Drift computation failed");
    } finally {
      setLoading(null);
    }
  }

  const confusion = result?.confusion_matrix;

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-6">
      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Train a Model</h2>
        <div className="flex flex-wrap items-center gap-2">
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value)}
            className="rounded border border-slate-300 px-3 py-1.5"
          >
            {TRAINABLE_ML_TASKS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-sm text-slate-600">
            Samples
            <input
              type="number"
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="w-24 rounded border border-slate-300 px-2 py-1"
            />
          </label>
          <label className="flex items-center gap-1 text-sm text-slate-600">
            Seed
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-20 rounded border border-slate-300 px-2 py-1"
            />
          </label>
          <button
            onClick={handleTrain}
            disabled={loading === "train"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 font-medium text-white disabled:opacity-50"
          >
            {loading === "train" ? "Training..." : "Train Model"}
          </button>
        </div>
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <>
          <div className="rounded-lg border border-slate-200 p-4">
            <h2 className="mb-2 text-lg font-semibold text-slate-900">
              Latest Model — {result.model.algorithm} ({result.duration_seconds.toFixed(3)}s, n={result.sample_size})
            </h2>
            <ul className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm text-slate-700 sm:grid-cols-3">
              {Object.entries(result.model.metrics).map(([name, value]) => (
                <li key={name}>
                  <span className="text-slate-500">{name}:</span> {value}
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg border border-slate-200 p-4">
            <h2 className="mb-2 text-lg font-semibold text-slate-900">Feature Importance</h2>
            <ResponsiveContainer width="100%" height={Math.max(180, result.feature_importance.length * 28)}>
              <BarChart data={result.feature_importance} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature_name" width={140} />
                <Tooltip />
                <Bar dataKey="importance" fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {result.roc_curve.length > 0 && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h2 className="mb-2 text-lg font-semibold text-slate-900">ROC Curve</h2>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={result.roc_curve}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottom", offset: -2 }} />
                  <YAxis dataKey="tpr" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="tpr" stroke="#059669" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {result.calibration_curve.length > 0 && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h2 className="mb-2 text-lg font-semibold text-slate-900">Calibration Curve</h2>
              <ResponsiveContainer width="100%" height={220}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="predicted_probability"
                    type="number"
                    domain={[0, 1]}
                    name="Predicted"
                    label={{ value: "Predicted probability", position: "insideBottom", offset: -2 }}
                  />
                  <YAxis
                    dataKey="observed_frequency"
                    type="number"
                    domain={[0, 1]}
                    name="Observed"
                    label={{ value: "Observed frequency", angle: -90, position: "insideLeft" }}
                  />
                  <Tooltip />
                  <Scatter data={result.calibration_curve} fill="#4f46e5" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}

          {confusion && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h2 className="mb-2 text-lg font-semibold text-slate-900">Confusion Matrix</h2>
              <table className="text-sm">
                <tbody>
                  {confusion.map((row, i) => (
                    <tr key={i}>
                      {row.map((cell, j) => (
                        <td key={j} className="border border-slate-200 px-4 py-2 text-center">
                          {cell}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}

      <div className="rounded-lg border border-slate-200 p-4">
        <div className="mb-2 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-900">Drift Monitoring</h2>
          <button
            onClick={handleDriftDemo}
            disabled={loading === "drift"}
            className="rounded-lg bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
          >
            {loading === "drift" ? "Computing..." : "Check Sample Drift"}
          </button>
        </div>
        {drift && (
          <p className="text-sm text-slate-700">
            PSI: <span className="font-medium">{drift.psi}</span> — severity:{" "}
            <span className="font-medium">{drift.severity}</span>
          </p>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Model Registry</h2>
        {models.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Model</th>
                <th className="pb-1">Task</th>
                <th className="pb-1">Algorithm</th>
                <th className="pb-1">Key Metric</th>
                <th className="pb-1">Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {models
                .slice()
                .reverse()
                .map((m) => (
                  <tr key={m.model_id} className="border-t border-slate-100">
                    <td className="py-1 font-mono text-xs">{m.model_id.slice(0, 8)}</td>
                    <td className="py-1">{m.task_type}</td>
                    <td className="py-1">{m.algorithm}</td>
                    <td className="py-1">
                      {Object.entries(m.metrics)
                        .slice(0, 1)
                        .map(([k, v]) => `${k}=${v}`)}
                    </td>
                    <td className="py-1 text-xs text-slate-500">{m.timestamp}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-slate-400">No models trained yet.</p>
        )}
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <h2 className="mb-2 text-lg font-semibold text-slate-900">Feature Store</h2>
        {featureDefs.length > 0 ? (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-slate-500">
                <th className="pb-1">Feature</th>
                <th className="pb-1">Description</th>
                <th className="pb-1">Source</th>
                <th className="pb-1">Type</th>
                <th className="pb-1">Version</th>
              </tr>
            </thead>
            <tbody>
              {featureDefs.map((f) => (
                <tr key={f.name} className="border-t border-slate-100">
                  <td className="py-1 font-medium">{f.name}</td>
                  <td className="py-1 text-slate-600">{f.description}</td>
                  <td className="py-1">{f.source}</td>
                  <td className="py-1">{f.datatype}</td>
                  <td className="py-1">{f.version}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="text-sm text-slate-400">No feature definitions found.</p>
        )}
      </div>
    </div>
  );
}
