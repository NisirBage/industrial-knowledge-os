def test_health_returns_ok(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_get_only(client):
    response = client.post("/api/v1/health")

    assert response.status_code == 405
