def _seed(client):
    client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "cc_seed.txt",
                b"Pump P-8901 failed due to Valve V-8902 blockage. "
                b"Pump P-8901 failed due to Valve V-8902 blockage again. "
                b"Motor M-8903 drives P-8901.",
                "text/plain",
            )
        },
    )


def test_preview_compression_returns_stats_and_evidence(client):
    _seed(client)

    response = client.post("/api/v1/context-compression/preview", json={"query": "Why did P-8901 fail?", "top_k": 10})

    assert response.status_code == 200
    body = response.json()
    assert "stats" in body
    assert body["stats"]["original_count"] >= 0
    assert isinstance(body["clusters"], list)
    assert isinstance(body["compressed_evidence"], list)


def test_preview_compression_rejects_unknown_level(client):
    response = client.post("/api/v1/context-compression/preview", json={"query": "test", "level": "not_a_real_level"})

    assert response.status_code == 422


def test_preview_compression_accepts_every_valid_level(client):
    _seed(client)
    for level in ("light", "balanced", "aggressive"):
        response = client.post("/api/v1/context-compression/preview", json={"query": "Why did P-8901 fail?", "level": level})
        assert response.status_code == 200, f"{level} failed: {response.text}"


def test_preview_compression_stats_are_internally_consistent(client):
    _seed(client)

    response = client.post("/api/v1/context-compression/preview", json={"query": "Why did P-8901 fail?", "top_k": 15})

    stats = response.json()["stats"]
    assert stats["compressed_count"] <= stats["original_count"]
    assert 0.0 <= stats["compression_ratio"] <= 1.0
    assert stats["summary"].startswith("Compressed")
