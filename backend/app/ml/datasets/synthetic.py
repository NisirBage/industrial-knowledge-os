"""Synthetic labeled datasets for training (M6).

Real historical failure/priority labels don't exist yet in this deployment
(that's exactly the gap M4's EquipmentFeatureVector already documents for
days_since_last_failure/average_repair_time/equipment_age). Rather than
train on random noise, these generators build a label from a documented
formula over the same feature schema plus Gaussian noise — the signal is
real and learnable (a trained model should beat a majority-class or
mean-value baseline), but explicitly synthetic, not a claim about actual
equipment behavior.

Only failure_prediction (classification) and maintenance_priority
(regression) are implemented — the two problem types Component 3 actually
trains — covering both concretely rather than fabricating labels for all
seven MLTaskType values. The other five follow the identical pattern
(TrainingDatasetBuilder.build with real labeled_equipment, or a synthetic
generator shaped like these two) once a real label source exists.
"""

import random

import pandas as pd

_MIN_FAILURE_COUNT, _MAX_FAILURE_COUNT = 0, 6
_MIN_INCIDENT_COUNT, _MAX_INCIDENT_COUNT = 0, 6
_MAX_FREQUENCY = 8


def _random_feature_row(rng: random.Random) -> dict[str, float]:
    failure_count = rng.randint(_MIN_FAILURE_COUNT, _MAX_FAILURE_COUNT)
    incident_count = rng.randint(_MIN_INCIDENT_COUNT, _MAX_INCIDENT_COUNT)
    contradiction_rate = round(rng.uniform(0, 0.6), 3)
    inspection_frequency = float(rng.randint(0, _MAX_FREQUENCY))
    maintenance_frequency = float(rng.randint(0, _MAX_FREQUENCY))
    risk_score = round(
        min(1.0, 0.4 * failure_count / 5 + 0.3 * incident_count / 5 + 0.3 * contradiction_rate), 4
    )
    return {
        "days_since_last_failure": rng.randint(0, 720),
        "maintenance_frequency": maintenance_frequency,
        "inspection_frequency": inspection_frequency,
        "failure_count": failure_count,
        "incident_count": incident_count,
        "average_repair_time": round(rng.uniform(1, 48), 2),
        "recommendation_acceptance_rate": round(rng.uniform(0, 1), 3),
        "contradiction_rate": contradiction_rate,
        "risk_score": risk_score,
        "equipment_age": rng.randint(0, 25),
        "ontology_depth": rng.randint(0, 2),
        "graph_degree": rng.randint(0, 20),
        "retrieval_frequency": rng.randint(0, 30),
    }


def generate_failure_prediction_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        row = _random_feature_row(rng)
        signal = (
            2.5 * row["risk_score"]
            + 0.4 * (row["failure_count"] / 5)
            + 0.3 * (row["incident_count"] / 5)
            - 0.3 * (row["inspection_frequency"] / _MAX_FREQUENCY)
            + rng.gauss(0, 0.6)
        )
        row["label"] = 1 if signal > 1.2 else 0
        rows.append(row)
    return pd.DataFrame(rows)


def generate_maintenance_priority_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_samples):
        row = _random_feature_row(rng)
        signal = (
            0.5 * row["risk_score"]
            + 0.3 * (1 - min(row["maintenance_frequency"] / _MAX_FREQUENCY, 1.0))
            + 0.2 * (row["failure_count"] / 5)
            + rng.gauss(0, 0.05)
        )
        row["label"] = round(max(0.0, min(1.0, signal)), 4)
        rows.append(row)
    return pd.DataFrame(rows)
