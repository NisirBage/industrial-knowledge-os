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

          {result.faithfulness && (
            <div className="rounded-lg border border-slate-200 p-4">
              <div className="mb-2 flex items-center justify-between">
                <h3 className="font-semibold text-slate-900">Faithfulness</h3>
                <span className="rounded-full bg-emerald-100 px-3 py-1 text-sm text-emerald-800">
                  {(result.faithfulness.overall_faithfulness * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-sm text-slate-600">{result.faithfulness.summary}</p>
              <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-4">
                <div>
                  <dt>Claim coverage</dt>
                  <dd className="font-medium text-slate-700">{(result.faithfulness.claim_coverage * 100).toFixed(0)}%</dd>
                </div>
                <div>
                  <dt>Graph faithfulness</dt>
                  <dd className="font-medium text-slate-700">{(result.faithfulness.graph_faithfulness * 100).toFixed(0)}%</dd>
                </div>
                <div>
                  <dt>Unsupported claims</dt>
                  <dd className="font-medium text-slate-700">{result.faithfulness.unsupported_claim_count}</dd>
                </div>
                <div>
                  <dt>Contradiction rate</dt>
                  <dd className="font-medium text-slate-700">{(result.faithfulness.contradiction_rate * 100).toFixed(0)}%</dd>
                </div>
              </dl>
            </div>
          )}

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
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs uppercase tracking-wide ${
                          r.verification_status === "verified"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-red-100 text-red-700"
                        }`}
                      >
                        {r.verification_status}
                      </span>
                    </div>
                    <p className="mt-1 text-slate-600">{r.justification}</p>
                    {r.verification_reasons.length > 0 && (
                      <p className="mt-1 text-xs text-slate-500">{r.verification_reasons.join(" ")}</p>
                    )}
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
                      {evidence?.explanation && (
                        <details className="mt-1">
                          <summary className="cursor-pointer text-xs text-indigo-600 hover:underline">Why was this retrieved?</summary>
                          <p className="mt-1 rounded bg-slate-50 p-2 text-xs text-slate-600">{evidence.explanation}</p>
                        </details>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {result.claims.length > 0 && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="mb-2 font-semibold text-slate-900">Claim Graph</h3>
              <ul className="flex flex-col gap-2">
                {result.claims.map((claim) => (
                  <li key={claim.id} className="text-sm text-slate-700">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs uppercase tracking-wide text-slate-600">
                        {claim.claim_type}
                      </span>
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs uppercase tracking-wide ${
                          claim.verification_status === "supported"
                            ? "bg-emerald-100 text-emerald-700"
                            : claim.verification_status === "partially_supported"
                              ? "bg-amber-100 text-amber-700"
                              : "bg-red-100 text-red-700"
                        }`}
                      >
                        {claim.verification_status}
                      </span>
                      <span className="text-slate-900">{claim.text}</span>
                    </div>
                    <p className="mt-0.5 text-xs text-slate-500">
                      confidence {claim.confidence.toFixed(2)}
                      {claim.propagated_confidence !== null && (
                        <> → propagated {claim.propagated_confidence.toFixed(2)}</>
                      )}
                      {" · "}
                      {claim.support_count} supporting source(s), {claim.contradiction_count} contradiction(s)
                    </p>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.sentence_attributions.length > 0 && (
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="mb-2 font-semibold text-slate-900">Reasoning Path</h3>
              <ul className="flex flex-col gap-2">
                {result.sentence_attributions.map((attribution, i) => (
                  <li key={i} className="text-sm text-slate-700">
                    <p>{attribution.sentence}</p>
                    <details className="mt-1">
                      <summary className="cursor-pointer text-xs text-indigo-600 hover:underline">Inspect reasoning path</summary>
                      <div className="mt-1 rounded bg-slate-50 p-2 text-xs text-slate-600">
                        <p>{attribution.reasoning_path}</p>
                        <p className="mt-1">confidence {attribution.confidence.toFixed(2)}</p>
                        {attribution.graph_edge_descriptions.length > 0 && (
                          <ul className="mt-1 list-inside list-disc">
                            {attribution.graph_edge_descriptions.map((edge, j) => (
                              <li key={j}>{edge}</li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </details>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
