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


def test_cache_stats_lists_every_named_retrieval_cache(client):
    response = client.get("/api/v1/observability/cache-stats")

    assert response.status_code == 200
    names = {c["name"] for c in response.json()}
    assert names == {"retrieval_embedding", "retrieval_expansion", "retrieval_reranker", "retrieval_result"}
    for entry in response.json():
        assert entry["hits"] >= 0
        assert entry["misses"] >= 0
        assert 0.0 <= entry["hit_ratio"] <= 1.0


def test_cache_stats_reflects_real_activity_from_a_repeated_chat_query(client):
    client.post("/api/v1/documents/upload", files={"file": ("seed.txt", b"Pump P-101 failed due to Valve V-202 blockage.", "text/plain")})

    client.post("/api/v1/chat", json={"query": "Why did P-101 fail?"})
    client.post("/api/v1/chat", json={"query": "Why did P-101 fail?"})  # identical repeat -> real cache hit

    response = client.get("/api/v1/observability/cache-stats")

    stats_by_name = {c["name"]: c for c in response.json()}
    assert stats_by_name["retrieval_result"]["hits"] >= 1
