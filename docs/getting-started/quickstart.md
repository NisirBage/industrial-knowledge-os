# Quick Start

This walks through the four workspaces using the [demo dataset](../../demo_data/README.md).
Run `python backend/scripts/seed_demo_data.py` first (see
[Installation](installation.md#seeding-demo-data)), or upload your own
document via **Documents**.

## 1. Chat

Open http://localhost:5173, stay on the **Chat** tab. Ask:

> What is the failure history of P-101?

You'll get a cited, evidence-backed answer — every claim traces back to a
specific ingested document chunk — plus a **Reasoning Summary** and any
**Contradictions** the Evidence Fusion Engine found across sources. This is the
8-stage LangGraph pipeline: Planner → Retrieval → Knowledge Graph → Evidence
Fusion → Reasoning → Recommendation → Verifier → Report. See
[docs/architecture/reasoning.md](../architecture/reasoning.md).

## 2. Documents

Switch to **Documents**. Upload a PDF, TXT, or MD file (or use one from
`demo_data/`). Ingestion runs the full pipeline synchronously: chunking →
embedding → entity extraction → relation extraction → knowledge graph update.
The response shows chunk/entity/relation counts immediately.

## 3. Research

Switch to **Research**. Run a benchmark (e.g. `entity_extraction`) against the
seeded synthetic dataset — this is the same [Research Framework](../architecture/research.md)
used to produce [docs/benchmarks/](../benchmarks). Every run creates a real,
reproducible experiment (see the **Experiment History** table at the bottom).

## 4. ML Lab

Switch to **ML Lab**. Click **Train Model** (task: `failure_prediction`). This
trains a real scikit-learn `RandomForestClassifier` on a synthetic labeled
dataset, in-process, in under 2 seconds — you'll see real accuracy/F1/ROC-AUC
metrics, feature importance, an ROC curve, a calibration curve, and a confusion
matrix, all computed from your own run (not canned numbers). The trained model
is registered and immediately swapped into the live Prediction Engine — the
next `/api/v1/predictions/{task}/{equipment_id}` call for `failure_risk` uses it.
See [docs/architecture/ml-lab.md](../architecture/ml-lab.md).

## What to try next

- Ask a question the ingested documents *don't* answer — see how IKIOS reports
  the gap rather than fabricating an answer.
- Upload two documents that disagree about the same equipment (e.g. conflicting
  valve states) and ask a question that touches both — the Contradiction
  Detector will flag it.
- Check `GET /api/v1/equipment/{equipment_id}/timeline?view=full` for the raw
  event timeline behind a Chat answer, or `POST /api/v1/predictions` with
  `{"equipment_id": "P-101", "task_type": "failure_risk"}` to call the
  Prediction Engine directly.
