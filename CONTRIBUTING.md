# Contributing to IKIOS

Thanks for your interest in contributing to the Industrial Knowledge Intelligence OS.

## Ground rules

- **No regressions.** Every milestone in this project's history has kept 100% of
  prior tests passing — that convention continues. Run the full suite before
  opening a PR.
- **Honest data handling.** Never fabricate metrics, labels, or confidence scores.
  If data is missing, model it as `None`/optional and say so — see
  [`EquipmentFeatureVector`](backend/app/features/models.py) for the pattern.
- **Graceful degradation.** Every heavy or optional dependency (GLiNER, Qdrant,
  Neo4j, cross-encoder reranker) must fall back to a documented, working
  alternative rather than crash the app. See `app/repositories/graph_repository.py`
  and `app/entity_extraction/factory.py` for the pattern.
- **Independent subsystems.** New milestones should avoid modifying the M3 chat
  orchestrator pipeline; expose new capability via its own FastAPI routes and
  `app.state` services instead. See [`docs/architecture/`](docs/architecture) for
  why.

## Development setup

See [Getting Started](docs/getting-started/installation.md) for the full setup, or
run `setup.ps1` / `setup.sh` from the repo root for a one-command bootstrap.

## Making a change

1. Fork the repo and create a branch off `main`.
2. Make your change. Match the existing style — no comments explaining *what* code
   does, only non-obvious *why*; dependency injection over globals; Protocol +
   concrete-implementation pairs for anything swappable (see any
   `*RepositoryProtocol` for the pattern).
3. Add tests. This project has no untested subsystem — new code should follow suit.
4. Run the full backend suite: `cd backend && pytest -q`. All tests must pass.
5. Run the frontend build: `cd frontend && npm run build`.
6. Open a pull request using the PR template. Describe *why*, not just *what*.

## Code style

- **Backend**: Python 3.13, type-hinted, `async`/`await` for I/O, dataclasses for
  internal models, Pydantic schemas only at the API boundary.
- **Frontend**: TypeScript, React function components, Tailwind utility classes,
  no CSS-in-JS.
- No premature abstraction — three similar lines beat a speculative helper.

## Reporting bugs / requesting features

Use the [issue templates](.github/ISSUE_TEMPLATE). Include repro steps, expected
vs. actual behavior, and which subsystem is affected (ingestion, knowledge graph,
retrieval, reasoning, learning, ML Lab, research, frontend, deployment).

## Questions

See [SUPPORT.md](SUPPORT.md).
