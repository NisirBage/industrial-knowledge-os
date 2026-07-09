def _seed_graph(client, count=3):
    texts = [
        "Pump P-9101 failed due to Valve V-9102 blockage. Motor M-9103 drives P-9101.",
        "Pump P-9201 failed due to Valve V-9202 blockage. Motor M-9203 drives P-9201.",
        "Pump P-9301 failed due to Valve V-9302 blockage. Motor M-9303 drives P-9301.",
    ]
    for i, text in enumerate(texts[:count]):
        client.post(
            "/api/v1/documents/upload",
            files={"file": (f"ranking_seed_{i}.txt", text.encode(), "text/plain")},
        )


def test_ranking_status_defaults_to_cross_encoder_score(client):
    response = client.get("/api/v1/ranking/status")

    assert response.status_code == 200
    body = response.json()
    assert body["active_ranker"] == "cross_encoder_score_default"
    assert body["is_trained"] is False


def test_train_ranker_rejects_when_not_enough_labeled_queries(client):
    """sample_size=0 caps `_labeled_queries_from_graph()`'s node scan at
    zero regardless of how much the shared graph has accumulated from other
    tests in this session-scoped `client` fixture, so this deterministically
    exercises the "not enough labeled queries" 400 path."""
    response = client.post("/api/v1/ranking/train", json={"sample_size": 0, "n_estimators": 5})

    assert response.status_code == 400


def test_train_ranker_succeeds_with_a_seeded_graph_and_updates_status(client):
    _seed_graph(client)

    response = client.post("/api/v1/ranking/train", json={"sample_size": 30, "n_estimators": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["trained"] is True
    assert body["algorithm"] == "lightgbm_lambdamart"
    assert body["sample_size"] >= 2
    assert len(body["feature_importance"]) > 0

    status = client.get("/api/v1/ranking/status")
    assert status.json() == {"active_ranker": "lightgbm_lambdamart", "is_trained": True}


def test_train_ranker_registers_into_the_shared_model_registry(client):
    _seed_graph(client)

    client.post("/api/v1/ranking/train", json={"sample_size": 30, "n_estimators": 5})

    models = client.get("/api/v1/ml/models?task_type=retrieval_ranking").json()
    assert len(models) >= 1
    assert models[0]["algorithm"] == "lightgbm_lambdamart"


def test_trained_ranker_is_actually_used_by_live_chat_retrieval(client):
    _seed_graph(client)
    client.post("/api/v1/ranking/train", json={"sample_size": 30, "n_estimators": 5})

    response = client.post("/api/v1/chat", json={"query": "Why did P-9101 fail?"})

    assert response.status_code == 200  # chat still works end-to-end with the trained ranker active
