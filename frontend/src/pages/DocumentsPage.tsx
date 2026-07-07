import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import { listDocuments, uploadDocument } from "../services/api";
import type { DocumentSummary } from "../services/api";

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUploadRelations, setLastUploadRelations] = useState<number | null>(null);

  async function refresh() {
    try {
      setDocuments(await listDocuments());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load documents");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      const result = await uploadDocument(file);
      setLastUploadRelations(result.num_relations);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-6">
      <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed border-slate-300 p-8 text-slate-500 hover:border-indigo-400">
        <span>{uploading ? "Ingesting..." : "Click to upload a PDF, TXT, or MD document"}</span>
        <input type="file" accept=".pdf,.txt,.md" className="hidden" onChange={handleFileChange} disabled={uploading} />
      </label>

      {error && <p className="rounded-lg bg-red-50 p-3 text-red-700">{error}</p>}
      {lastUploadRelations !== null && (
        <p className="rounded-lg bg-indigo-50 p-3 text-sm text-indigo-700">
          Last upload extracted {lastUploadRelations} graph relationship{lastUploadRelations === 1 ? "" : "s"}.
        </p>
      )}

      <div className="flex flex-col gap-2">
        {documents.map((doc) => (
          <div key={doc.id} className="flex items-center justify-between rounded-lg border border-slate-200 p-3">
            <div>
              <p className="font-medium text-slate-900">{doc.filename}</p>
              <p className="text-sm text-slate-500">{doc.content_type}</p>
            </div>
            <div className="text-sm text-slate-600">
              {doc.num_chunks} chunks · {doc.num_entities} entities
            </div>
          </div>
        ))}
        {documents.length === 0 && <p className="text-center text-slate-400">No documents ingested yet.</p>}
      </div>
    </div>
  );
}
