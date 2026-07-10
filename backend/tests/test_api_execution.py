"""O3.3 API tests: a real chat turn persists an ExecutionTrace (via
ChatService's optional execution_repository), then the read-side
/api/v1/executions endpoints expose it — list, lookup, timeline
reconstruction, aggregate statistics, and failure events."""


def _seed_and_chat(client, tag: str) -> str:
    client.post(
        "/api/v1/documents/upload",
        files={"file": (f"exec_seed_{tag}.txt", f"Pump {tag} failed due to a bearing fault.".encode(), "text/plain")},
    )
    response = client.post("/api/v1/chat", json={"query": f"Why did {tag} fail?"})
    assert response.status_code == 200
    return response.json()["summary"]


def test_chat_turn_produces_a_listed_execution_trace(client):
    _seed_and_chat(client, "P-7001")

    response = client.get("/api/v1/executions")
    assert response.status_code == 200
    traces = response.json()
    assert len(traces) >= 1
    assert all("execution_id" in t for t in traces)
    assert all(len(t["events"]) > 0 for t in traces)


def test_get_execution_by_id_returns_full_trace(client):
    _seed_and_chat(client, "P-7002")

    execution_id = client.get("/api/v1/executions").json()[0]["execution_id"]
    response = client.get(f"/api/v1/executions/{execution_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["execution_id"] == execution_id
    assert "planning_started" in {e["event_type"] for e in body["events"]}


def test_get_execution_returns_404_for_unknown_id(client):
    response = client.get("/api/v1/executions/nonexistent-execution-id")
    assert response.status_code == 404


def test_get_execution_timeline_returns_reconstructed_dag(client):
    _seed_and_chat(client, "P-7003")

    execution_id = client.get("/api/v1/executions").json()[0]["execution_id"]
    response = client.get(f"/api/v1/executions/{execution_id}/timeline")

    assert response.status_code == 200
    body = response.json()
    assert body["component_order"]
    assert body["dag"]["nodes"] == body["component_order"]
    assert len(body["dag"]["edges"]) == len(body["dag"]["nodes"]) - 1


def test_execution_statistics_reflect_recorded_traces(client):
    _seed_and_chat(client, "P-7004")

    response = client.get("/api/v1/executions/statistics")
    assert response.status_code == 200
    body = response.json()
    assert body["trace_count"] >= 1
    assert body["event_count"] > 0
    assert isinstance(body["average_latency_by_component"], dict)


def test_list_failure_events_returns_empty_list_when_none_recorded(client):
    response = client.get("/api/v1/executions/failures/list")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
