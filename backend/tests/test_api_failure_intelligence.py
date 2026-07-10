"""O3.4 API tests: mining/clustering/root-cause-graph/counterfactual/
optimization-recommendation endpoints over ExecutionTraces real chat turns
already persisted (see test_api_execution.py for the underlying trace
persistence path)."""


def _seed_and_chat(client, tag: str) -> None:
    client.post(
        "/api/v1/documents/upload",
        files={"file": (f"fi_seed_{tag}.txt", f"Pump {tag} failed due to a bearing fault.".encode(), "text/plain")},
    )
    response = client.post("/api/v1/chat", json={"query": f"Why did {tag} fail?"})
    assert response.status_code == 200


def test_mine_patterns_returns_a_list(client):
    _seed_and_chat(client, "P-9101")
    _seed_and_chat(client, "P-9102")

    response = client.post("/api/v1/failure-intelligence/patterns/mine")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_list_patterns_reflects_mined_patterns(client):
    _seed_and_chat(client, "P-9103")
    client.post("/api/v1/failure-intelligence/patterns/mine")

    response = client.get("/api/v1/failure-intelligence/patterns")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_clusters_rule_based_by_category(client):
    _seed_and_chat(client, "P-9104")

    response = client.get("/api/v1/failure-intelligence/clusters", params={"strategy": "rule_based", "key": "component_sequence"})
    assert response.status_code == 200
    clusters = response.json()
    assert isinstance(clusters, list)
    if clusters:
        assert "execution_ids" in clusters[0]


def test_get_clusters_rejects_unknown_strategy(client):
    response = client.get("/api/v1/failure-intelligence/clusters", params={"strategy": "not_a_real_strategy"})
    assert response.status_code == 400


def test_get_root_cause_graph_for_a_real_execution(client):
    _seed_and_chat(client, "P-9105")

    execution_id = client.get("/api/v1/executions").json()[0]["execution_id"]
    response = client.get(f"/api/v1/failure-intelligence/root-cause-graph/{execution_id}")

    assert response.status_code == 200
    body = response.json()
    assert any(n["type"] == "execution" for n in body["nodes"])


def test_get_root_cause_graph_returns_404_for_unknown_execution(client):
    response = client.get("/api/v1/failure-intelligence/root-cause-graph/nonexistent")
    assert response.status_code == 404


def test_simulate_counterfactual_on_a_real_execution(client):
    _seed_and_chat(client, "P-9106")

    execution_id = client.get("/api/v1/executions").json()[0]["execution_id"]
    response = client.post(f"/api/v1/failure-intelligence/counterfactual/{execution_id}", json={"toggle": "verification"})

    assert response.status_code == 200
    body = response.json()
    assert body["execution_id"] == execution_id
    assert body["toggle"] == "verification"


def test_simulate_counterfactual_rejects_unknown_toggle(client):
    _seed_and_chat(client, "P-9107")
    execution_id = client.get("/api/v1/executions").json()[0]["execution_id"]

    response = client.post(f"/api/v1/failure-intelligence/counterfactual/{execution_id}", json={"toggle": "not_a_real_toggle"})
    assert response.status_code == 422


def test_simulate_counterfactual_returns_404_for_unknown_execution(client):
    response = client.post("/api/v1/failure-intelligence/counterfactual/nonexistent", json={"toggle": "verification"})
    assert response.status_code == 404


def test_get_optimization_recommendations_returns_a_list(client):
    response = client.get("/api/v1/failure-intelligence/optimization-recommendations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
