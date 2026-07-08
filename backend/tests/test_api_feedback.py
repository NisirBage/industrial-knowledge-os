def test_accept_recommendation_feedback_happy_path(client):
    response = client.post(
        "/api/v1/feedback",
        json={"kind": "recommendation_accepted", "target_id": "rec-1", "equipment_id": "P-9001"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "recommendation_accepted"
    assert body["target_id"] == "rec-1"


def test_reject_recommendation_feedback_happy_path(client):
    response = client.post("/api/v1/feedback", json={"kind": "recommendation_rejected", "target_id": "rec-2"})

    assert response.status_code == 200


def test_unknown_feedback_kind_returns_422_with_valid_options(client):
    response = client.post("/api/v1/feedback", json={"kind": "not_a_real_kind", "target_id": "x"})

    assert response.status_code == 422
    assert "entity_correction" in response.json()["detail"]


def test_entity_correction_without_corrected_value_returns_422(client):
    response = client.post("/api/v1/feedback", json={"kind": "entity_correction", "target_id": "entity-1"})

    assert response.status_code == 422
    assert "corrected_value" in response.json()["detail"]


def test_entity_correction_with_corrected_value_succeeds(client):
    response = client.post(
        "/api/v1/feedback",
        json={"kind": "entity_correction", "target_id": "entity-1", "corrected_value": "Pump"},
    )

    assert response.status_code == 200
    assert response.json()["corrected_value"] == "Pump"


def test_feedback_missing_required_fields_returns_422(client):
    response = client.post("/api/v1/feedback", json={"kind": "recommendation_accepted"})

    assert response.status_code == 422
