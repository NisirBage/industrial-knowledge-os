def _seed(client):
    client.post(
        "/api/v1/documents/upload",
        files={"file": ("er_seed.txt", b"Pump P-8801 failed due to Valve V-8802 blockage. Motor M-8803 drives P-8801.", "text/plain")},
    )


def test_find_duplicates_returns_a_list(client):
    _seed(client)

    response = client.get("/api/v1/entity-resolution/duplicates?threshold=0.82")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_find_duplicates_with_a_near_impossible_threshold_returns_no_matches(client):
    """Uses threshold=0.999 rather than an empty graph: `client` is a
    session-scoped fixture shared with every other test in the suite, so
    the graph may already contain nodes seeded by earlier tests by the time
    this runs — a threshold no ordinary similarity score can clear is a
    session-order-independent way to assert "no duplicates found"."""
    _seed(client)

    response = client.get("/api/v1/entity-resolution/duplicates?threshold=0.999")

    assert response.status_code == 200
    assert response.json() == []


def test_merge_two_real_nodes_and_verify_graph_state(client):
    _seed(client)
    summary = client.get("/api/v1/graph-intelligence/summary").json()
    assert summary["node_count"] > 0

    graph = client.get("/api/v1/graph-intelligence/graph").json()
    pump_nodes = [n for n in graph["nodes"] if n["type"] == "Pump"]
    assert len(pump_nodes) >= 1
    other_node = next(n for n in graph["nodes"] if n["id"] != pump_nodes[0]["id"])

    response = client.post(
        "/api/v1/entity-resolution/merge",
        json={"source_id": pump_nodes[0]["id"], "target_id": other_node["id"], "reason": "test_merge"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reason"] == "test_merge"
    assert "merge_id" in body

    after = client.get("/api/v1/graph-intelligence/graph").json()
    node_ids_after = {n["id"] for n in after["nodes"]}
    assert pump_nodes[0]["id"] not in node_ids_after


def test_merge_unknown_node_returns_400(client):
    response = client.post(
        "/api/v1/entity-resolution/merge",
        json={"source_id": "does-not-exist", "target_id": "also-does-not-exist", "reason": "test"},
    )

    assert response.status_code == 400


def test_merge_then_rollback_restores_the_node(client):
    _seed(client)
    graph = client.get("/api/v1/graph-intelligence/graph").json()
    source_node, target_node = graph["nodes"][0], graph["nodes"][1]

    merge_response = client.post(
        "/api/v1/entity-resolution/merge",
        json={"source_id": source_node["id"], "target_id": target_node["id"], "reason": "test"},
    )
    merge_id = merge_response.json()["merge_id"]

    rollback_response = client.post("/api/v1/entity-resolution/rollback", json={"merge_id": merge_id})

    assert rollback_response.status_code == 200
    assert rollback_response.json()["restored_node_id"] == source_node["id"]

    after = client.get("/api/v1/graph-intelligence/graph").json()
    node_ids_after = {n["id"] for n in after["nodes"]}
    assert source_node["id"] in node_ids_after


def test_rollback_unknown_merge_id_returns_404(client):
    response = client.post("/api/v1/entity-resolution/rollback", json={"merge_id": "does-not-exist"})

    assert response.status_code == 404
