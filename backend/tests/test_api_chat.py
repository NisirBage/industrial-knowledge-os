def test_chat_happy_path_returns_cited_answer(client):
    response = client.post("/api/v1/chat", json={"query": "What is the failure history of P-101?"})

    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "confidence" in body
    assert isinstance(body["citations"], list)
    assert isinstance(body["contradictions"], list)
    assert isinstance(body["recommendations"], list)
    assert "latencies" in body


def test_chat_missing_query_field_returns_422(client):
    response = client.post("/api/v1/chat", json={})

    assert response.status_code == 422


def test_chat_rejects_non_json_body(client):
    response = client.post("/api/v1/chat", data="not json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_chat_accepts_empty_string_query(client):
    response = client.post("/api/v1/chat", json={"query": ""})

    assert response.status_code == 200


def test_chat_rejects_get(client):
    response = client.get("/api/v1/chat")

    assert response.status_code == 405
