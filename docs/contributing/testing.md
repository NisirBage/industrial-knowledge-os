# Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -q                                        # all 301 tests
pytest -q --cov=app --cov-report=term-missing     # with coverage
pytest -q tests/test_ml_trainer.py                # one file
```

## How the suite is organized

One test file per subsystem, named after the module it covers
(`test_ml_trainer.py` ↔ `app/ml/training/trainer.py`). No shared
`conftest.py` — each test file is self-contained with small local fixture
functions (e.g. `_feature_vector()`, `_registry(tmp_path)`).

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
- **Route handlers are verified live, not via `TestClient`.** Every
  milestone's engines/repositories are unit-tested exhaustively; the FastAPI
  route layer itself is verified by starting the real server and driving it
  from a browser/HTTP client during that milestone's release process (see
  each milestone's section in [docs/RELEASE_REPORT.md](../RELEASE_REPORT.md)).

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
