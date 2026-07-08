"""Generates the small, high-quality vision demo dataset (M7) into
../vision_demo_data/ — deterministic, programmatically drawn images with
EXACT known ground truth (bounding boxes, OCR text, gauge angle/value,
defect type/severity), the same "seeded synthetic data with real,
controllable ground truth" approach the text demo dataset and the M5
Synthetic Dataset Generator both use, since no real industrial photos are
available to this project.

Usage:
    cd backend
    python scripts/generate_vision_demo_data.py
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np

_OUT_DIR = Path(__file__).resolve().parent.parent.parent / "vision_demo_data"

_GAUGE_CENTER = (200, 200)
_GAUGE_RADIUS = 120
_GAUGE_NEEDLE_ANGLE_DEGREES = 40.0  # -135=min(0%), 0=up(50%), +135=max(100%)


def _needle_endpoint(center, radius, angle_degrees, length_fraction=0.75):
    length = radius * length_fraction
    angle_rad = math.radians(angle_degrees)
    dx = length * math.sin(angle_rad)
    dy = -length * math.cos(angle_rad)
    return int(center[0] + dx), int(center[1] + dy)


def generate_gauge_image() -> dict:
    img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.circle(img, _GAUGE_CENTER, _GAUGE_RADIUS, (0, 0, 0), 4)
    endpoint = _needle_endpoint(_GAUGE_CENTER, _GAUGE_RADIUS, _GAUGE_NEEDLE_ANGLE_DEGREES)
    cv2.line(img, _GAUGE_CENTER, endpoint, (0, 0, 0), 5)
    cv2.putText(img, "PRESSURE GAUGE PG-201", (60, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    path = _OUT_DIR / "gauge_pg201.png"
    cv2.imwrite(str(path), img)

    expected_value = (_GAUGE_NEEDLE_ANGLE_DEGREES - (-135.0)) / 270.0
    return {
        "file": path.name,
        "source_type": "equipment_photo",
        "description": "Pressure gauge PG-201 with needle at a known angle.",
        "ground_truth": {
            "gauge_center": list(_GAUGE_CENTER),
            "gauge_radius": _GAUGE_RADIUS,
            "needle_angle_degrees": _GAUGE_NEEDLE_ANGLE_DEGREES,
            "expected_normalized_value": round(expected_value, 4),
            "ocr_text": "PRESSURE GAUGE PG-201",
        },
    }


def generate_pid_diagram() -> dict:
    img = np.full((400, 500, 3), 255, dtype=np.uint8)
    valve_center, valve_radius = (100, 100), 40
    cv2.circle(img, valve_center, valve_radius, (0, 0, 0), 3)
    pump_box = (250, 60, 420, 160)
    cv2.rectangle(img, pump_box[:2], pump_box[2:], (0, 0, 0), 3)
    cv2.line(img, (140, 100), (250, 100), (0, 0, 0), 3)  # pipe connecting valve -> pump
    cv2.putText(img, "V-2201A", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "P-101", (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    path = _OUT_DIR / "pid_diagram.png"
    cv2.imwrite(str(path), img)

    return {
        "file": path.name,
        "source_type": "pid_diagram",
        "description": "Simple P&ID: valve V-2201A connected by a pipe to pump P-101.",
        "ground_truth": {
            "valve_bounding_box_approx": [valve_center[0] - valve_radius, valve_center[1] - valve_radius, valve_center[0] + valve_radius, valve_center[1] + valve_radius],
            "pump_bounding_box_approx": list(pump_box),
            "ocr_text": ["V-2201A", "P-101"],
            "symbols_expected": ["valve", "pump", "pipe"],
        },
    }


def generate_equipment_photo_with_corrosion() -> dict:
    img = np.full((300, 300, 3), 210, dtype=np.uint8)
    equipment_box = (70, 70, 230, 230)
    cv2.rectangle(img, equipment_box[:2], equipment_box[2:], (60, 60, 60), -1)
    corrosion_box = (150, 150, 210, 210)
    cv2.rectangle(img, corrosion_box[:2], corrosion_box[2:], (14, 65, 183), -1)  # BGR rust color
    cv2.putText(img, "M-500", (95, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    path = _OUT_DIR / "equipment_m500_corrosion.png"
    cv2.imwrite(str(path), img)

    return {
        "file": path.name,
        "source_type": "inspection_photo",
        "description": "Motor M-500 with a visible corrosion patch on its housing.",
        "ground_truth": {
            "equipment_bounding_box_approx": list(equipment_box),
            "defect_bounding_box_approx": list(corrosion_box),
            "defect_type": "corrosion",
            "ocr_text": "M-500",
        },
    }


def generate_crack_inspection_photo() -> dict:
    img = np.full((250, 350, 3), 225, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (320, 220), (150, 150, 150), -1)
    cv2.line(img, (60, 190), (300, 205), (20, 20, 20), 2)  # thin elongated crack
    cv2.putText(img, "TANK T-300", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    path = _OUT_DIR / "tank_t300_crack.png"
    cv2.imwrite(str(path), img)

    return {
        "file": path.name,
        "source_type": "inspection_photo",
        "description": "Tank T-300 with a thin surface crack.",
        "ground_truth": {
            "defect_type": "crack",
            "ocr_text": "TANK T-300",
        },
    }


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = [
        generate_gauge_image(),
        generate_pid_diagram(),
        generate_equipment_photo_with_corrosion(),
        generate_crack_inspection_photo(),
    ]
    ground_truth = {
        "description": (
            "Hand-designed, deterministic, small vision demo dataset (M7) with exact "
            "known ground truth — analogous to demo_data/ground_truth.json for text. "
            "No real industrial photographs were available to this project; these are "
            "programmatically drawn scenes with controllable, verifiable ground truth."
        ),
        "images": entries,
    }
    (_OUT_DIR / "ground_truth.json").write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")
    print(f"Wrote {len(entries)} images + ground_truth.json to {_OUT_DIR}")


if __name__ == "__main__":
    main()
