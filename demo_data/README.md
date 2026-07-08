# Demo Dataset

A small, coherent industrial scenario — Pump P-101, its upstream isolation
Valve V-2201A, and a realistic bearing-failure incident — spanning all 7
curated document types, specifically built to exercise every subsystem:

| File | Document type | Exercises |
| --- | --- | --- |
| `maintenance_log_P101.txt` | Maintenance log | Ingestion, entity extraction, timeline |
| `inspection_report_P101.txt` | Inspection report | Compliance/standard entities, **contradiction seed #1** (valve OPEN) |
| `incident_report_P101.txt` | Incident report | Root cause reasoning, **contradiction seed #2** (valve CLOSED), recommendation triggers |
| `oem_manual_P101.txt` | OEM manual | Failure-mode knowledge, standard reference |
| `sop_P101.txt` | SOP | Procedure/permit relations |
| `equipment_history_P101.txt` | Equipment history | Full timeline reconstruction, related-equipment graph edges |
| `failure_report_P101.txt` | Failure report | Recommendation grounding, cost/downtime data |

`ground_truth.json` is the hand-labeled reference: expected entities,
relations, one deliberate cross-document contradiction (Valve V-2201A's state
disagrees between the inspection and incident reports — a realistic
"undocumented handoff between crews" scenario, not a contrived test fixture),
sample questions with expected-answer keywords, expected recommendations, and
the expected *direction* (not a fabricated exact number) for
failure-risk/maintenance-priority predictions.

## Using it

```bash
cd backend
python scripts/seed_demo_data.py
```

Requires the backend to be running (`http://localhost:8000` by default —
override with `IKIOS_API_BASE_URL`). Uploads all 7 files via
`POST /api/v1/documents/upload`, the same path a real user upload takes — no
special seeding backdoor.

Then try the questions in `ground_truth.json`, or follow
[docs/getting-started/quickstart.md](../docs/getting-started/quickstart.md).

## Why this dataset, specifically

Every other synthetic dataset in this repo
([`app/research/datasets/generator.py`](../backend/app/research/datasets/generator.py))
is randomly generated per-seed for benchmarking at scale. This one is
hand-written and fixed on purpose — a demo needs the *same* story every time
so a reviewer can compare what they see against `ground_truth.json` directly.
