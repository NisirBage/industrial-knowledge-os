import asyncio
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.ml.registry.repository import SqlModelRegistry, hash_dataframe


async def _registry(tmp_path) -> SqlModelRegistry:
    db_path = tmp_path / "ml_registry_test.db"
    models_dir = tmp_path / "models"
    registry = SqlModelRegistry(f"sqlite+aiosqlite:///{db_path}", models_dir)
    await registry.init_models()
    return registry


def _fitted_estimator() -> RandomForestClassifier:
    estimator = RandomForestClassifier(n_estimators=10, random_state=1)
    estimator.fit([[0, 0], [1, 1], [0, 1], [1, 0]], [0, 1, 1, 0])
    return estimator


async def test_register_creates_a_retrievable_record(tmp_path):
    registry = await _registry(tmp_path)

    record = await registry.register(
        estimator=_fitted_estimator(),
        task_type="failure_prediction",
        algorithm="RandomForestClassifier",
        training_data_hash="abc123",
        hyperparameters={"n_estimators": "10"},
        metrics={"accuracy": 0.9},
        feature_schema=["risk_score", "failure_count"],
        author="test-suite",
    )

    fetched = await registry.get(record.model_id)
    assert fetched is not None
    assert fetched.task_type == "failure_prediction"
    assert fetched.metrics == {"accuracy": 0.9}
    assert fetched.feature_schema == ["risk_score", "failure_count"]
    assert fetched.serialization_format == "joblib"
    assert Path(fetched.serialization_path).exists()


async def test_get_returns_none_for_unknown_model_id(tmp_path):
    registry = await _registry(tmp_path)
    assert await registry.get("nonexistent-id") is None


async def test_load_estimator_round_trips_via_joblib(tmp_path):
    registry = await _registry(tmp_path)
    estimator = _fitted_estimator()

    record = await registry.register(
        estimator=estimator,
        task_type="failure_prediction",
        algorithm="RandomForestClassifier",
        training_data_hash="abc123",
        hyperparameters={},
        metrics={},
        feature_schema=["a", "b"],
    )

    loaded = await registry.load_estimator(record.model_id)
    assert list(loaded.predict([[0, 0], [1, 1]])) == list(estimator.predict([[0, 0], [1, 1]]))


async def test_get_latest_returns_most_recently_registered_for_task_type(tmp_path):
    registry = await _registry(tmp_path)

    await registry.register(
        estimator=_fitted_estimator(),
        task_type="failure_prediction",
        algorithm="RandomForestClassifier",
        training_data_hash="v1",
        hyperparameters={},
        metrics={"accuracy": 0.5},
        feature_schema=["a"],
    )
    await asyncio.sleep(0.01)
    second = await registry.register(
        estimator=_fitted_estimator(),
        task_type="failure_prediction",
        algorithm="RandomForestClassifier",
        training_data_hash="v2",
        hyperparameters={},
        metrics={"accuracy": 0.9},
        feature_schema=["a"],
    )

    latest = await registry.get_latest("failure_prediction")
    assert latest is not None
    assert latest.model_id == second.model_id


async def test_get_latest_returns_none_for_unknown_task_type(tmp_path):
    registry = await _registry(tmp_path)
    assert await registry.get_latest("nonexistent_task") is None


async def test_list_filters_by_task_type(tmp_path):
    registry = await _registry(tmp_path)

    await registry.register(
        estimator=_fitted_estimator(),
        task_type="failure_prediction",
        algorithm="RandomForestClassifier",
        training_data_hash="v1",
        hyperparameters={},
        metrics={},
        feature_schema=["a"],
    )
    await registry.register(
        estimator=_fitted_estimator(),
        task_type="maintenance_priority",
        algorithm="RandomForestRegressor",
        training_data_hash="v1",
        hyperparameters={},
        metrics={},
        feature_schema=["a"],
    )

    all_records = await registry.list()
    failure_records = await registry.list("failure_prediction")

    assert len(all_records) == 2
    assert len(failure_records) == 1
    assert failure_records[0].task_type == "failure_prediction"


def test_hash_dataframe_is_deterministic_for_identical_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert hash_dataframe(df) == hash_dataframe(df.copy())


def test_hash_dataframe_differs_for_different_data():
    df_a = pd.DataFrame({"a": [1, 2, 3]})
    df_b = pd.DataFrame({"a": [1, 2, 4]})
    assert hash_dataframe(df_a) != hash_dataframe(df_b)
