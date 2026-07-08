def test_prediction_happy_path_failure_risk(client):
    response = client.post("/api/v1/predictions", json={"equipment_id": "P-9001", "task_type": "failure_risk"})

    assert response.status_code == 200
    body = response.json()
    assert body["equipment_id"] == "P-9001"
    assert body["task_type"] == "failure_risk"
    assert "confidence" in body
    assert "explanation" in body


def test_prediction_unknown_task_type_returns_422_with_valid_options_listed(client):
    response = client.post("/api/v1/predictions", json={"equipment_id": "P-9001", "task_type": "not_a_real_task"})

    assert response.status_code == 422
    assert "failure_risk" in response.json()["detail"]


def test_prediction_missing_equipment_id_returns_422(client):
    response = client.post("/api/v1/predictions", json={"task_type": "failure_risk"})

    assert response.status_code == 422


def test_prediction_every_documented_task_type_is_reachable(client):
    task_types = [
        "remaining_useful_life",
        "failure_risk",
        "inspection_priority",
        "maintenance_priority",
        "recommendation_priority",
        "missing_documentation_risk",
        "compliance_risk",
    ]
    for task_type in task_types:
        response = client.post("/api/v1/predictions", json={"equipment_id": "P-9002", "task_type": task_type})
        assert response.status_code == 200, f"{task_type} failed: {response.text}"
