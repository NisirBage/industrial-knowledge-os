# Adding a New Subsystem

Every milestone since M4 (Learning Layer) has followed the same shape. This is
what let M4, M5, and M6 each add substantial new capability without ever
touching the M3 chat orchestrator or risking its 50+ tests.

## The pattern

1. **New package under `app/`** (e.g. `app/ml/`) — data models as
   `@dataclass`, business logic as plain classes, no framework coupling.
2. **Protocol + implementation** for anything persistence-backed, matching
   the existing repositories (`GraphRepositoryProtocol`,
   `ModelRegistryProtocol`, ...).
3. **New Pydantic schemas** in `app/schemas/your_subsystem.py` — the only
   place your dataclasses get serialized.
4. **New router** in `app/api/v1/your_subsystem.py`, included in
   [`app/api/v1/router.py`](../../backend/app/api/v1/router.py).
5. **Wire into `app/main.py`'s `lifespan()`** — construct your services once,
   attach to `app.state`. Reuse existing `app.state` services where your
   subsystem needs the same underlying data (e.g. M6 reused M4's
   `feature_extractor` and M5's `experiment_manager`) instead of building a
   parallel implementation.
6. **Tests first-class, not an afterthought** — see
   [testing.md](testing.md).
7. **Frontend**: a new page in `frontend/src/pages/`, a new tab in `App.tsx`,
   new types + fetch functions in `services/api.ts`. Never touches existing
   pages.
8. **Docs**: an architecture doc under `docs/architecture/`, an entry in
   [docs/api/README.md](../api/README.md).

## Why this matters

The chat orchestrator (M3) is the most complex, highest-risk piece of the
platform — an 8-stage LangGraph pipeline with contradiction detection and
evidence fusion. Every subsequent milestone deliberately avoided touching it,
not because it's fragile, but because *not needing to* is itself validation
that the layering is right. If a new subsystem ever seems to require changing
the orchestrator, that's a signal to reconsider the design before writing
code, not a step to take lightly.

## Runtime extension point: swapping a component without a new subsystem

Not every capability needs a new subsystem — `PredictionEngine.register_model()`
is the pattern for swapping one *implementation* of an existing interface at
runtime (a trained ML model replacing a rule-based one). Reach for this first
if you're improving something that already has a Protocol, and reach for a
full new subsystem only when the capability genuinely doesn't fit an existing
one.
