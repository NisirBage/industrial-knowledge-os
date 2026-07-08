from app.ml.datasets.synthetic import generate_failure_prediction_dataset, generate_maintenance_priority_dataset

_EXPECTED_FEATURE_COLUMNS = {
    "days_since_last_failure",
    "maintenance_frequency",
    "inspection_frequency",
    "failure_count",
    "incident_count",
    "average_repair_time",
    "recommendation_acceptance_rate",
    "contradiction_rate",
    "risk_score",
    "equipment_age",
    "ontology_depth",
    "graph_degree",
    "retrieval_frequency",
}


def test_failure_prediction_dataset_has_expected_shape_and_columns():
    df = generate_failure_prediction_dataset(200, seed=1)

    assert len(df) == 200
    assert _EXPECTED_FEATURE_COLUMNS.issubset(set(df.columns))
    assert set(df["label"].unique()).issubset({0, 1})


def test_failure_prediction_dataset_is_deterministic_for_same_seed():
    df_a = generate_failure_prediction_dataset(100, seed=42)
    df_b = generate_failure_prediction_dataset(100, seed=42)

    assert df_a.equals(df_b)


def test_failure_prediction_dataset_differs_across_seeds():
    df_a = generate_failure_prediction_dataset(100, seed=1)
    df_b = generate_failure_prediction_dataset(100, seed=2)

    assert not df_a.equals(df_b)


def test_failure_prediction_dataset_has_both_classes_present():
    df = generate_failure_prediction_dataset(300, seed=42)
    assert set(df["label"].unique()) == {0, 1}


def test_failure_prediction_signal_correlates_with_risk_score():
    df = generate_failure_prediction_dataset(500, seed=42)
    high_risk_rate = df[df["risk_score"] > df["risk_score"].median()]["label"].mean()
    low_risk_rate = df[df["risk_score"] <= df["risk_score"].median()]["label"].mean()

    assert high_risk_rate > low_risk_rate


def test_maintenance_priority_dataset_has_expected_shape_and_columns():
    df = generate_maintenance_priority_dataset(150, seed=3)

    assert len(df) == 150
    assert _EXPECTED_FEATURE_COLUMNS.issubset(set(df.columns))
    assert df["label"].between(0.0, 1.0).all()


def test_maintenance_priority_dataset_is_deterministic_for_same_seed():
    df_a = generate_maintenance_priority_dataset(80, seed=9)
    df_b = generate_maintenance_priority_dataset(80, seed=9)

    assert df_a.equals(df_b)


def test_maintenance_priority_dataset_label_is_continuous():
    df = generate_maintenance_priority_dataset(100, seed=9)
    assert df["label"].nunique() > 10
