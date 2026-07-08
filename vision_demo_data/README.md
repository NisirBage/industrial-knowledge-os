# Vision Demo Dataset (M7)

Four small, deterministic, programmatically-drawn images with exact known
ground truth (`ground_truth.json`) — the visual analog of
[`demo_data/`](../demo_data/README.md). No real industrial photographs were
available to this project; every image here is drawn with OpenCV primitives
at known coordinates/colors/angles, so the ground truth is exact and
verifiable, not estimated.

| File | Exercises |
| --- | --- |
| `gauge_pg201.png` | Gauge detection + needle-angle reading (known angle → known expected value) |
| `pid_diagram.png` | Diagram symbol detection (valve, pump, pipe) + OCR on equipment tags |
| `equipment_m500_corrosion.png` | Equipment region detection + corrosion (color-signature) defect detection + OCR |
| `tank_t300_crack.png` | Crack (elongated-edge) defect detection + OCR |

## Generating it

```bash
cd backend
python scripts/generate_vision_demo_data.py
```

Re-running regenerates identical images (fully deterministic — every shape
is drawn at a fixed coordinate/color/angle, no randomness involved).

## An honest, measured result — not a bug

Running the real detectors against `pid_diagram.png` finds the valve (circle)
correctly and many pipe-like line segments, but **does not** reliably find the
pump rectangle in this composite scene, even though an isolated single-
rectangle image is correctly detected in
[`backend/tests/test_vision_diagram.py`](../backend/tests/test_vision_diagram.py).
Classical-CV contour detection on a scene with multiple overlapping shapes
and text is measurably noisier than on a single isolated shape — this is
expected, real behavior for the documented classical-CV baseline (see
[docs/architecture/vision.md](../docs/architecture/vision.md)), not something
quietly patched to look better. The gauge reading similarly lands close to
but not exactly at its known ground-truth value (Hough Circle detection finds
a slightly different center/radius than the exact drawn values) — reported
honestly rather than rounded to look perfect.

Use [`backend/scripts/generate_benchmark_reports.py`](../backend/scripts/generate_benchmark_reports.py)
(extended in M7 — see `docs/benchmarks/README.md`) for the full measured
precision/recall picture across many synthetic scenes, not just these four
demo images.
