# FAQ

**Is this just a RAG chatbot?**
No. Retrieval is one of seven layers. IKIOS builds a persistent knowledge
graph, reasons over evidence with explicit contradiction detection and a
reasoning graph, generates justified recommendations, learns from feedback,
and trains real scikit-learn models — a chatbot is the surface, not the
architecture. See [docs/architecture/system-overview.md](../architecture/system-overview.md).

**Do I need Docker / Neo4j / Qdrant to run this?**
No. Every one of them is optional — the app falls back automatically and logs
a warning. See [docs/architecture/deployment.md](../architecture/deployment.md#fallback-matrix).

**Do I need an Anthropic API key?**
No. `LLM_PROVIDER=mock` (the default) gives deterministic, offline responses
for every stage except final narration. Reasoning, contradiction detection,
retrieval, and predictions all work identically either way.

**Does IKIOS do OCR / read scanned PDFs?**
Not yet. Today's ingestion supports text-layer PDFs (via `pdfplumber`), TXT,
and MD. Scanned/image-only documents, DOCX, and XLSX are documented future
work — see [docs/architecture/ingestion.md](../architecture/ingestion.md#supported-formats-today).
This is intentionally stated plainly rather than implied by the vision
document: the platform never claims a capability it doesn't have.

**Is the ML in ML Lab real, or is it a mockup?**
Real. `POST /api/v1/ml/train/{task}` fits an actual scikit-learn
`RandomForestClassifier`/`Regressor` on a dataset, computes real evaluation
metrics (accuracy, F1, ROC-AUC, calibration, etc.) on a genuinely held-out
test split, and the trained model is hot-swapped into the live Prediction
Engine. The training *labels* are synthetic (documented, seeded, formula-
derived) because real historical failure labels don't exist in this
deployment yet — the model, training, and evaluation are not.

**Why SQLite instead of Postgres?**
An MVP substitution, documented from the start (see
[CLAUDE.md](../../CLAUDE.md)) — every repository is behind a `Protocol`, so
swapping to Postgres is a new implementation + config change, not a rewrite.
See [docs/roadmap/README.md](../roadmap/README.md).

**Why do M4/M5/M6 never touch the M3 orchestrator?**
Deliberate risk management — the orchestrator is the most complex, highest-
stakes part of the platform. Every later milestone proves its own value
without ever risking regressing it. See
[docs/contributing/adding-a-subsystem.md](../contributing/adding-a-subsystem.md).

**How many tests are there, and do they actually all pass?**
301, and yes — every milestone's test suite is a strict superset of the
previous one's; nothing has ever been weakened or deleted to make a later
milestone's tests pass. Run `pytest -q` from `backend/` yourself.

**Can I use a different LLM provider?**
`app/services/llm_provider.py` defines an `LLMProvider` Protocol with `Mock`
and `Anthropic` implementations. Adding OpenAI/Gemini/etc. is a new class
implementing the same Protocol, wired in via `build_llm_provider()`.

**Where do I report a bug or ask something not covered here?**
See [SUPPORT.md](../../SUPPORT.md).
