def test_timeline_default_view_returns_full_history(client):
    response = client.get("/api/v1/equipment/P-9001/timeline")

    assert response.status_code == 200
    body = response.json()
    assert body["equipment"] == "P-9001"
    assert body["view"] == "full"
    assert isinstance(body["events"], list)


def test_timeline_accepts_each_documented_view(client):
    for view in ["full", "equipment", "incident", "maintenance"]:
        response = client.get(f"/api/v1/equipment/P-9001/timeline?view={view}")
        assert response.status_code == 200
        assert response.json()["view"] == view


def test_timeline_rejects_unknown_view(client):
    response = client.get("/api/v1/equipment/P-9001/timeline?view=bogus")

    assert response.status_code == 422


def test_timeline_for_unknown_equipment_returns_empty_events_not_an_error(client):
    response = client.get("/api/v1/equipment/DOES-NOT-EXIST-99999/timeline")

    assert response.status_code == 200
    assert response.json()["events"] == []
