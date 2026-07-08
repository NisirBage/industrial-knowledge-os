def test_run_each_auto_runnable_benchmark(client):
    for benchmark_name in ["entity_extraction", "relation_extraction", "knowledge_graph"]:
        response = client.post(f"/api/v1/research/benchmarks/{benchmark_name}/run", json={"sample_size": 5, "seed": 1})
        assert response.status_code == 200, f"{benchmark_name} failed: {response.text}"
        body = response.json()
        assert body["benchmark_name"] == benchmark_name
        assert isinstance(body["metrics"], list)


def test_run_each_vision_benchmark(client):
    vision_benchmarks = [
        "vision_ocr",
        "vision_diagram_parsing",
        "vision_equipment_recognition",
        "vision_gauge_reading",
        "vision_defect_detection",
        "vision_multimodal_retrieval",
        "vision_latency",
    ]
    for benchmark_name in vision_benchmarks:
        response = client.post(f"/api/v1/research/benchmarks/{benchmark_name}/run", json={"sample_size": 5, "seed": 1})
        assert response.status_code == 200, f"{benchmark_name} failed: {response.text}"
        body = response.json()
        assert body["benchmark_name"] == benchmark_name
        assert isinstance(body["metrics"], list)
        assert len(body["metrics"]) > 0


def test_run_each_graph_benchmark(client):
    graph_benchmarks = [
        "graph_embedding_node2vec",
        "graph_embedding_deepwalk",
        "graph_embedding_gcn",
        "graph_embedding_graphsage",
        "graph_embedding_gat",
        "graph_link_prediction",
        "graph_node_classification",
    ]
    for benchmark_name in graph_benchmarks:
        response = client.post(f"/api/v1/research/benchmarks/{benchmark_name}/run", json={"sample_size": 5, "seed": 1})
        assert response.status_code == 200, f"{benchmark_name} failed: {response.text}"
        body = response.json()
        assert body["benchmark_name"] == benchmark_name
        assert isinstance(body["metrics"], list)
        assert len(body["metrics"]) > 0


def test_graph_benchmark_run_appears_in_history(client):
    client.post("/api/v1/research/benchmarks/graph_link_prediction/run", json={"sample_size": 5, "seed": 4})

    response = client.get("/api/v1/research/benchmarks/graph_link_prediction/history")

    assert response.status_code == 200
    assert all(e["name"] == "benchmark:graph_link_prediction" for e in response.json())


def test_vision_benchmark_run_appears_in_history(client):
    client.post("/api/v1/research/benchmarks/vision_gauge_reading/run", json={"sample_size": 5, "seed": 4})

    response = client.get("/api/v1/research/benchmarks/vision_gauge_reading/history")

    assert response.status_code == 200
    assert all(e["name"] == "benchmark:vision_gauge_reading" for e in response.json())


def test_run_non_auto_runnable_benchmark_returns_400(client):
    response = client.post("/api/v1/research/benchmarks/hybrid_retrieval/run", json={"sample_size": 5, "seed": 1})

    assert response.status_code == 400
    assert "hybrid_retrieval" in response.json()["detail"]


def test_experiments_list_includes_completed_benchmark_run(client):
    client.post("/api/v1/research/benchmarks/entity_extraction/run", json={"sample_size": 5, "seed": 2})

    response = client.get("/api/v1/research/experiments")

    assert response.status_code == 200
    names = {e["name"] for e in response.json()}
    assert "benchmark:entity_extraction" in names


def test_get_unknown_experiment_returns_404(client):
    response = client.get("/api/v1/research/experiments/does-not-exist")

    assert response.status_code == 404


def test_benchmark_history_filters_by_name(client):
    client.post("/api/v1/research/benchmarks/knowledge_graph/run", json={"sample_size": 5, "seed": 3})

    response = client.get("/api/v1/research/benchmarks/knowledge_graph/history")

    assert response.status_code == 200
    assert all(e["name"] == "benchmark:knowledge_graph" for e in response.json())


def test_entity_extraction_ab_test_happy_path(client):
    response = client.post("/api/v1/research/ab-test/entity-extraction", json={"sample_size": 5, "seed": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["arm_a_name"] == "regex_only"
    assert 0.0 <= body["p_value"] <= 1.0


def test_calibration_returns_buckets_list(client):
    response = client.get("/api/v1/research/calibration")

    assert response.status_code == 200
    assert isinstance(response.json()["buckets"], list)


def test_research_benchmark_run_missing_body_uses_defaults(client):
    response = client.post("/api/v1/research/benchmarks/entity_extraction/run", json={})

    assert response.status_code == 200
