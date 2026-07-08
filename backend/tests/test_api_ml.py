def test_train_failure_prediction_happy_path(client):
    response = client.post("/api/v1/ml/train/failure_prediction", json={"sample_size": 60, "seed": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["algorithm"] == "RandomForestClassifier"
    assert "accuracy" in body["model"]["metrics"]
    assert isinstance(body["feature_importance"], list)


def test_train_without_cross_validation_by_default(client):
    response = client.post("/api/v1/ml/train/failure_prediction", json={"sample_size": 60, "seed": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["cross_validation_metrics"] is None
    assert body["cross_validation_fold_count"] is None


def test_train_with_cross_validation_opt_in(client):
    response = client.post(
        "/api/v1/ml/train/failure_prediction",
        json={"sample_size": 60, "seed": 1, "include_cross_validation": True, "cv_folds": 3},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["cross_validation_fold_count"] == 3
    assert "accuracy" in body["cross_validation_metrics"]
    # Distinct from the held-out point estimate — both present, clearly labeled.
    assert "accuracy" in body["model"]["metrics"]


def test_train_maintenance_priority_happy_path(client):
    response = client.post("/api/v1/ml/train/maintenance_priority", json={"sample_size": 60, "seed": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["algorithm"] == "RandomForestRegressor"
    assert "rmse" in body["model"]["metrics"]


def test_train_unknown_task_type_returns_422(client):
    response = client.post("/api/v1/ml/train/not_a_real_task", json={"sample_size": 60, "seed": 1})

    assert response.status_code == 422


def test_train_not_yet_trainable_task_returns_400(client):
    response = client.post("/api/v1/ml/train/root_cause_classification", json={"sample_size": 60, "seed": 1})

    assert response.status_code == 400
    assert "root_cause_classification" in response.json()["detail"]


def test_train_missing_body_uses_defaults(client):
    response = client.post("/api/v1/ml/train/failure_prediction", json={})

    assert response.status_code == 200


def test_models_list_includes_trained_model(client):
    trained = client.post("/api/v1/ml/train/failure_prediction", json={"sample_size": 60, "seed": 5})
    model_id = trained.json()["model"]["model_id"]

    response = client.get("/api/v1/ml/models")

    assert response.status_code == 200
    ids = {m["model_id"] for m in response.json()}
    assert model_id in ids


def test_models_list_filters_by_task_type(client):
    client.post("/api/v1/ml/train/failure_prediction", json={"sample_size": 60, "seed": 6})

    response = client.get("/api/v1/ml/models?task_type=failure_prediction")

    assert response.status_code == 200
    assert all(m["task_type"] == "failure_prediction" for m in response.json())


def test_get_model_by_id_happy_path(client):
    trained = client.post("/api/v1/ml/train/failure_prediction", json={"sample_size": 60, "seed": 7})
    model_id = trained.json()["model"]["model_id"]

    response = client.get(f"/api/v1/ml/models/{model_id}")

    assert response.status_code == 200
    assert response.json()["model_id"] == model_id


def test_get_unknown_model_id_returns_404(client):
    response = client.get("/api/v1/ml/models/does-not-exist")

    assert response.status_code == 404


def test_feature_definitions_returns_sixteen_features(client):
    response = client.get("/api/v1/ml/feature-definitions")

    assert response.status_code == 200
    assert len(response.json()) == 16


def test_drift_endpoint_happy_path(client):
    response = client.post(
        "/api/v1/ml/drift",
        json={"baseline": [0.1, 0.2, 0.3, 0.4, 0.5] * 10, "recent": [0.5, 0.6, 0.7, 0.8, 0.9] * 10},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["severity"] in {"none", "moderate", "significant"}


def test_drift_endpoint_missing_fields_returns_422(client):
    response = client.post("/api/v1/ml/drift", json={"baseline": [0.1, 0.2]})

    assert response.status_code == 422
