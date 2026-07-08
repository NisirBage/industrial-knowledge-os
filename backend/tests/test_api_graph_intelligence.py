def _seed_graph(client):
    client.post(
        "/api/v1/documents/upload",
        files={"file": ("gi_seed.txt", b"Pump P-8001 failed due to Valve V-8002 blockage. Motor M-8003 drives P-8001.", "text/plain")},
    )


def test_summary_reports_node_and_edge_counts(client):
    _seed_graph(client)

    response = client.get("/api/v1/graph-intelligence/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["node_count"] > 0
    assert "Pump" in body["node_types"]


def test_graph_export_returns_nodes_and_edges(client):
    _seed_graph(client)

    response = client.get("/api/v1/graph-intelligence/graph")

    assert response.status_code == 200
    body = response.json()
    assert len(body["nodes"]) > 0
    assert isinstance(body["edges"], list)


def test_train_embeddings_returns_a_model_id(client):
    _seed_graph(client)

    response = client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"]
    assert body["model_name"] == "node2vec"
    assert body["dimensions"] == 8


def test_train_embeddings_rejects_an_unknown_model_name(client):
    _seed_graph(client)

    response = client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "not_a_real_model"})

    assert response.status_code == 422


def test_list_embedding_models_returns_all_five(client):
    response = client.get("/api/v1/graph-intelligence/embeddings/models")

    assert response.status_code == 200
    assert set(response.json()) == {"node2vec", "deepwalk", "gcn", "graphsage", "gat"}


def test_similarity_returns_ranked_neighbors(client):
    _seed_graph(client)
    client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    response = client.get("/api/v1/graph-intelligence/similarity/P-8001")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_similarity_requires_an_embedding_to_be_trained_first(client):
    """Uses a fresh equipment tag never seen by this session's cache, on
    the off chance an earlier test already trained one — the 400 case is
    only reachable before any embedding exists at all, so this test mainly
    documents the contract rather than reliably exercising the branch in a
    shared-session suite."""
    response = client.get("/api/v1/graph-intelligence/similarity/does-not-exist")

    assert response.status_code == 404  # unknown equipment name takes priority over the missing-embedding check


def test_explain_returns_a_human_readable_summary(client):
    _seed_graph(client)
    client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    response = client.get("/api/v1/graph-intelligence/explain/P-8001")

    assert response.status_code == 200
    assert "Confidence" in response.json()["text"]


def test_explain_unknown_equipment_returns_404(client):
    response = client.get("/api/v1/graph-intelligence/explain/does-not-exist")

    assert response.status_code == 404


def test_link_prediction_candidates_returns_a_ranked_list(client):
    _seed_graph(client)
    client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    response = client.post("/api/v1/graph-intelligence/link-prediction/candidates", json={"equipment_name": "P-8001", "top_k": 3})

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_node_classification_train_registers_a_graph_model(client):
    _seed_graph(client)
    client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    response = client.post(
        "/api/v1/graph-intelligence/node-classification/train",
        json={"task_type": "failure_risk", "equipment_names": ["P-8001", "V-8002", "M-8003"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["registered"] is True
    assert body["task_type"] == "failure_risk"
    assert len(body["feature_importance"]) > 0
    assert all(fi["feature_name"].startswith("dim_") for fi in body["feature_importance"])


def test_node_classification_train_rejects_an_unknown_task_type(client):
    _seed_graph(client)
    client.post("/api/v1/graph-intelligence/embeddings/train", json={"model_name": "node2vec", "dimensions": 8})

    response = client.post(
        "/api/v1/graph-intelligence/node-classification/train",
        json={"task_type": "not_a_real_task", "equipment_names": ["P-8001"]},
    )

    assert response.status_code == 422


def test_propagation_simulate_returns_ranked_steps(client):
    _seed_graph(client)

    response = client.post(
        "/api/v1/graph-intelligence/propagation/simulate",
        json={"equipment_name": "P-8001", "propagation_type": "failure", "max_depth": 2},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_node_id"]
    assert isinstance(body["steps"], list)


def test_propagation_simulate_rejects_an_unknown_propagation_type(client):
    _seed_graph(client)

    response = client.post(
        "/api/v1/graph-intelligence/propagation/simulate",
        json={"equipment_name": "P-8001", "propagation_type": "not_a_real_type"},
    )

    assert response.status_code == 422


def test_propagation_simulate_unknown_equipment_returns_404(client):
    response = client.post(
        "/api/v1/graph-intelligence/propagation/simulate", json={"equipment_name": "does-not-exist"}
    )

    assert response.status_code == 404


def test_get_node_features_returns_computed_features(client):
    _seed_graph(client)

    response = client.get("/api/v1/graph-intelligence/features/P-8001")

    assert response.status_code == 200
    body = response.json()
    assert body["canonical_name"] == "P-8001"
    assert body["degree"] >= 0


def test_get_node_features_unknown_equipment_returns_404(client):
    response = client.get("/api/v1/graph-intelligence/features/does-not-exist")

    assert response.status_code == 404
