def test_observability_metrics_shape(client):
    response = client.get("/api/v1/observability/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "feedback_volume" in body
    assert "average_prediction_confidence" in body
    assert "average_prediction_latency_seconds" in body
    assert "average_feature_generation_latency_seconds" in body


def test_observability_metrics_reflects_prior_predictions(client):
    client.post("/api/v1/predictions", json={"equipment_id": "P-OBS-1", "task_type": "failure_risk"})

    response = client.get("/api/v1/observability/metrics")

    assert response.status_code == 200
    assert response.json()["average_prediction_confidence"] is not None
