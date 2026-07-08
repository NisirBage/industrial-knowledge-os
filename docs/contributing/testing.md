# Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q                                        # all tests
pytest -q --cov=app --cov-report=term-missing     # with coverage
pytest -q tests/test_ml_trainer.py                # one file
pytest -q tests/test_api_ml.py                    # one API-layer file
```

## How the suite is organized

One test file per subsystem, named after the module it covers
(`test_ml_trainer.py` ↔ `app/ml/training/trainer.py`). Business-logic test
files are self-contained with small local fixture functions (e.g.
`_feature_vector()`, `_registry(tmp_path)`) and share no `conftest.py`.

## API-layer tests (v1.0.1)

`tests/test_api_*.py` (one file per router) use a shared, **session-scoped**
`client` fixture in `tests/conftest.py` — a real `TestClient` wrapping the
actual `create_app()`, running the full composition-root `lifespan()` exactly
once for the whole test session, against temporary/ephemeral storage (never
a developer's real `ikios_*.db` files). This is what closed the "zero
automated tests exercise the HTTP layer" gap identified in the v1.0.0 audit:
every route now has at least one happy-path test and one 4xx-path test,
verifying request validation, status codes, and response shape — not just the
business logic underneath, which the rest of the suite already covered
thoroughly.

Because the fixture is session-scoped, state accumulates across API test
files within one run (a document uploaded in `test_api_documents.py` is
visible to a later test in the same session) — write new API tests to assert
on what *they* create or on qualitative response shape, not on exact global
counts that other tests could affect.

This ordering dependency is real, not theoretical: an early version of
`test_api_vision.py` had a test that queried `/vision/query` before any
document had ever been ingested in that run, which crashed with a raw `500`
(the vector store's collection doesn't exist until the first upload calls
`ensure_collection()`) — but only when that test file was run in isolation,
since other `test_api_*.py` files alphabetically before it already upload
documents when the full suite runs. The real fix was in the app itself
(`MultimodalRetriever.retrieve()` now degrades gracefully instead of
crashing — see [docs/architecture/multimodal-retrieval.md](../architecture/multimodal-retrieval.md#graceful-degradation)),
but the lesson for test authors: if a test's pass/fail depends on what ran
before it in the same file, that's a sign the code under test has a real gap,
not just a test-ordering inconvenience to work around.

## Vision tests (M7)

`test_vision_*.py` (ingestion, OCR, diagram, detection, gauge, defect, graph
builder, multimodal retrieval, abstractions, benchmarks) are business-logic
tests following the same self-contained-fixture convention as the rest of
the suite. `test_vision_ocr.py` exercises the real EasyOCR model in a
module-scoped fixture, with `pytest.skip()` if this environment's
intermittent native-library (torch DLL) loading flakiness occurs — the same
pattern already used for GLiNER. `test_api_vision.py` follows the shared
`client` fixture convention above.

## Frontend tests (M7.1)

The backend test suite above has always been comprehensive; the frontend had
**zero** automated tests before M7.1. `frontend/src/pages/VisionPage.test.tsx`
and `frontend/src/services/api.vision.test.ts` are the first, using Vitest +
React Testing Library + `@testing-library/user-event` (installed from
scratch — see `frontend/vitest.config.ts` and `frontend/src/test/setup.ts`).

```bash
cd frontend
npm test              # vitest run
npm run test:watch    # vitest, watch mode
npm run test:coverage # vitest run --coverage
```

`services/api.*.test.ts` files mock `global.fetch` directly (no MSW/nock) —
consistent with `api.ts` itself being plain `fetch` calls, not axios.
`pages/*.test.tsx` files mock `../services/api` module functions via
`vi.mock` with `importOriginal` (keeping the real exported constants like
`VISION_SOURCE_TYPES` intact, only mocking the async functions), then drive
the component with `@testing-library/user-event` and assert on rendered
text/roles — never on internal component state.

## Patterns used throughout

- **Real SQLite via `tmp_path`**, not mocks, for anything backed by
  SQLAlchemy — `SqlLearningRepository`, `SqlExperimentTracker`,
  `SqlModelRegistry` are all tested against a real (temporary) database file.
- **Hand-written fakes implementing the same Protocol**, not a mocking
  framework — see `_FakeFeatureExtractor`, `_RecordingLearningRepository` in
  `tests/test_prediction_engine.py` for the pattern.
- **Deterministic seeded data** wherever randomness is involved
  (`random.Random(seed)`) — every synthetic dataset test asserts identical
  output for the same seed and different output for a different seed.
- **No mocking of scikit-learn** — training tests fit real estimators on real
  (synthetic) data and assert on real computed metrics, including a
  "beats a naive baseline" check, not just "didn't crash."
- **Route handlers are covered by `TestClient` tests (`test_api_*.py`,
  added in v1.0.1)** in addition to live manual verification during each
  milestone's release process (see each milestone's section in
  [docs/RELEASE_REPORT.md](../RELEASE_REPORT.md)) — engines/repositories are
  unit-tested exhaustively, and the HTTP boundary now has automated
  regression coverage too.

## The one rule that matters most

**No regressions.** Every milestone's test count is a strict superset of the
previous one's. If your change breaks an existing test, fix the break — don't
weaken or delete the test to make it pass.

## Adding tests for a new subsystem

1. One test file, named after the module.
2. Cover: the happy path, at least one edge case (empty input, unknown
   enum value, missing optional field), and — if the module does anything
   numeric — a check against a known/derivable value, not just "no exception."
3. Run the full suite before opening a PR, not just your new file.
