import { useState } from "react";
import type { FormEvent } from "react";
import { sendChatQuery } from "../services/api";
import type { ChatResponse } from "../services/api";

export function ChatPage() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await sendChatQuery(query);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6">
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask about equipment, maintenance, compliance..."
          className="flex-1 rounded-lg border border-slate-300 px-4 py-2 focus:border-indigo-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white disabled:opacity-50"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="rounded-lg border border-slate-200 p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-900">Answer</h2>
              <span className="rounded-full bg-slate-100 px-3 py-1 text-sm text-slate-600">
                confidence {(result.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <p className="whitespace-pre-wrap text-slate-700">{result.summary}</p>
          </div>

          {result.reasoning_summary && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <h3 className="mb-1 font-semibold text-slate-900">Reasoning Summary</h3>
              <p className="text-sm text-slate-600">{result.reasoning_summary}</p>
            </div>
          )}

          {result.contradictions.length > 0 && (
            <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
              <h3 className="mb-2 font-semibold text-amber-900">Contradictions</h3>
              <ul className="flex flex-col gap-2">
                {result.contradictions.map((c, i) => (
                  <li key={i} className="text-sm text-amber-800">
                    <span className="rounded bg-amber-100 px-1.5 py-0.5 text-xs uppercase tracking-wide">{c.severity}</span>{" "}
                    {c.recommended_clarification}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.recommendations.length > 0 ? (
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="mb-2 font-semibold text-slate-900">Recommendations</h3>
              <ul className="flex flex-col gap-3">
                {result.recommendations.map((r) => (
                  <li key={r.id} className="text-sm text-slate-700">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-indigo-100 px-1.5 py-0.5 text-xs uppercase tracking-wide text-indigo-700">
                        {r.priority}
                      </span>
                      <span className="font-medium text-slate-900">{r.description}</span>
                    </div>
                    <p className="mt-1 text-slate-600">{r.justification}</p>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            result.recommended_actions.length > 0 && (
              <div className="rounded-lg border border-slate-200 p-4">
                <h3 className="mb-2 font-semibold text-slate-900">Recommended Actions</h3>
                <ul className="list-inside list-disc text-slate-700">
                  {result.recommended_actions.map((action, i) => (
                    <li key={i}>{action}</li>
                  ))}
                </ul>
              </div>
            )
          )}

          {result.citations.length > 0 && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="mb-2 font-semibold text-slate-900">Sources</h3>
              <ul className="flex flex-col gap-2">
                {result.citations.map((citation) => {
                  const evidence = result.evidence.find((e) => e.chunk_id === citation.chunk_id);
                  return (
                    <li key={citation.chunk_id} className="text-sm text-slate-600">
                      <span className="font-medium text-slate-900">[{citation.index}]</span>{" "}
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs uppercase tracking-wide text-slate-500">
                        {citation.source}
                      </span>{" "}
                      {citation.source === "graph"
                        ? "knowledge graph relationship"
                        : `document ${citation.document_id} — page ${citation.page_number}`}
                      {evidence && <p className="mt-1 italic text-slate-500">{evidence.text.slice(0, 200)}...</p>}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
