import pandas as pd

from app.features.models import EquipmentFeatureVector
from app.ml.datasets.builder import TrainingDatasetBuilder


def _feature_vector(equipment_id: str, risk_score: float) -> EquipmentFeatureVector:
    return EquipmentFeatureVector(
        equipment_id=equipment_id,
        days_since_last_failure=10,
        maintenance_frequency=2.0,
        inspection_frequency=1.0,
        failure_count=1,
        incident_count=1,
        average_repair_time=None,
        recommendation_acceptance_rate=None,
        contradiction_rate=0.1,
        risk_score=risk_score,
        equipment_age=None,
        manufacturer=None,
        location=None,
        criticality="medium",
        ontology_depth=1,
        graph_degree=3,
        retrieval_frequency=0,
    )


class _FakeFeatureExtractor:
    async def extract(self, equipment_id: str) -> EquipmentFeatureVector:
        return _feature_vector(equipment_id, risk_score=len(equipment_id) / 10)


async def test_build_produces_one_row_per_labeled_equipment():
    builder = TrainingDatasetBuilder(_FakeFeatureExtractor())
    df = await builder.build({"P-101": 1.0, "P-102": 0.0})

    assert len(df) == 2
    assert set(df["equipment_id"]) == {"P-101", "P-102"}
    assert "risk_score" in df.columns


async def test_build_attaches_label_column_with_custom_name():
    builder = TrainingDatasetBuilder(_FakeFeatureExtractor())
    df = await builder.build({"P-101": 0.75}, label_column="priority")

    assert df.loc[0, "priority"] == 0.75
    assert "priority" in df.columns


def _sample_dataframe(n: int = 50) -> pd.DataFrame:
    return pd.DataFrame({"feature_a": range(n), "feature_b": [i * 2 for i in range(n)], "label": [i % 2 for i in range(n)]})


def test_train_test_split_respects_test_size():
    df = _sample_dataframe(100)
    train, test = TrainingDatasetBuilder.train_test_split(df, test_size=0.2)

    assert len(train) == 80
    assert len(test) == 20
    assert set(train.columns) == set(df.columns)


def test_train_test_split_is_deterministic_for_fixed_seed():
    df = _sample_dataframe(50)
    train_a, test_a = TrainingDatasetBuilder.train_test_split(df, random_state=7)
    train_b, test_b = TrainingDatasetBuilder.train_test_split(df, random_state=7)

    assert list(train_a.index) == list(train_b.index)
    assert list(test_a.index) == list(test_b.index)


def test_train_test_split_stratify_preserves_label_ratio():
    df = pd.DataFrame({"feature_a": range(100), "label": [0] * 80 + [1] * 20})
    _, test = TrainingDatasetBuilder.train_test_split(df, test_size=0.5, stratify=True)

    assert test["label"].sum() == 10


def test_time_series_split_covers_all_rows_across_folds():
    df = _sample_dataframe(30)
    folds = TrainingDatasetBuilder.time_series_split(df, n_splits=3)

    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert max(train_idx) < min(test_idx)


def test_cross_validation_folds_partition_all_rows():
    df = _sample_dataframe(20)
    folds = TrainingDatasetBuilder.cross_validation_folds(df, n_splits=4)

    assert len(folds) == 4
    for train_idx, test_idx in folds:
        assert len(train_idx) + len(test_idx) == 20
        assert set(train_idx).isdisjoint(set(test_idx))


def test_to_csv_round_trips_dataframe(tmp_path):
    df = _sample_dataframe(10)
    path = tmp_path / "dataset.csv"

    TrainingDatasetBuilder.to_csv(df, path)
    reloaded = pd.read_csv(path)

    assert list(reloaded["feature_a"]) == list(df["feature_a"])


def test_to_parquet_round_trips_dataframe(tmp_path):
    df = _sample_dataframe(10)
    path = tmp_path / "dataset.parquet"

    TrainingDatasetBuilder.to_parquet(df, path)
    reloaded = pd.read_parquet(path)

    assert list(reloaded["feature_b"]) == list(df["feature_b"])
