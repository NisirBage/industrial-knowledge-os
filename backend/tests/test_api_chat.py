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


def test_chat_evidence_includes_explainability_fields(client):
    client.post(
        "/api/v1/documents/upload",
        files={"file": ("explain_seed.txt", b"Pump P-9501 failed due to Valve V-9502 blockage.", "text/plain")},
    )

    response = client.post("/api/v1/chat", json={"query": "Why did P-9501 fail?"})

    assert response.status_code == 200
    evidence = response.json()["evidence"]
    assert len(evidence) > 0
    for i, item in enumerate(evidence, start=1):
        assert "ltr_score" in item
        assert item["final_rank"] == i
        assert item["explanation"].startswith(f"Rank #{i}.")
