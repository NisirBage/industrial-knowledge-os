from io import BytesIO

from PIL import Image, ImageDraw


def _png_bytes(size=(64, 64), color=(200, 200, 200)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _png_bytes_with_object(size=(200, 200)) -> bytes:
    """A background with one high-contrast rectangle — enough foreground
    signal for the real Otsu-threshold object detector to find a region,
    unlike a flat-color image (which yields zero contours)."""
    image = Image.new("RGB", size, (210, 210, 210))
    ImageDraw.Draw(image).rectangle([50, 50, 150, 150], fill=(20, 20, 20))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_image_returns_document_with_metadata(client):
    response = client.post(
        "/api/v1/vision/images",
        files={"file": ("gauge.png", _png_bytes(), "image/png")},
        data={"source_type": "equipment_photo", "equipment_hint": "PG-900"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "gauge.png"
    assert body["source_type"] == "equipment_photo"
    assert body["equipment_hint"] == "PG-900"
    assert body["width"] == 64 and body["height"] == 64


def test_uploaded_image_appears_in_list(client):
    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("list_check.png", _png_bytes(), "image/png")},
        data={"source_type": "inspection_photo"},
    )
    image_id = upload.json()["id"]

    listing = client.get("/api/v1/vision/images")

    assert listing.status_code == 200
    ids = {d["id"] for d in listing.json()}
    assert image_id in ids


def test_get_image_by_id_returns_document(client):
    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("get_check.png", _png_bytes(), "image/png")},
        data={"source_type": "pid_diagram"},
    )
    image_id = upload.json()["id"]

    response = client.get(f"/api/v1/vision/images/{image_id}")

    assert response.status_code == 200
    assert response.json()["id"] == image_id


def test_get_missing_image_returns_404(client):
    response = client.get("/api/v1/vision/images/does-not-exist")

    assert response.status_code == 404


def test_get_image_file_returns_the_raw_bytes(client):
    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("file_check.png", _png_bytes(), "image/png")},
        data={"source_type": "inspection_photo"},
    )
    image_id = upload.json()["id"]

    response = client.get(f"/api/v1/vision/images/{image_id}/file")

    assert response.status_code == 200
    assert response.content == _png_bytes()
    assert response.headers["content-type"] == "image/png"


def test_get_missing_image_file_returns_404(client):
    response = client.get("/api/v1/vision/images/does-not-exist/file")

    assert response.status_code == 404


def test_unsupported_source_type_returns_422(client):
    response = client.post(
        "/api/v1/vision/images",
        files={"file": ("bad.png", _png_bytes(), "image/png")},
        data={"source_type": "not_a_real_source_type"},
    )

    assert response.status_code == 422


def test_invalid_image_bytes_returns_422(client):
    response = client.post(
        "/api/v1/vision/images",
        files={"file": ("bad.png", b"not actually a png", "image/png")},
        data={"source_type": "inspection_photo"},
    )

    assert response.status_code == 422


def test_analyze_image_returns_analysis_with_all_result_fields(client):
    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("analyze_check.png", _png_bytes(), "image/png")},
        data={"source_type": "inspection_photo"},
    )
    image_id = upload.json()["id"]

    response = client.post(f"/api/v1/vision/images/{image_id}/analyze")

    assert response.status_code == 200
    body = response.json()
    assert body["image_id"] == image_id
    for key in ("ocr_regions", "diagram_symbols", "detections", "gauge_readings", "defects"):
        assert key in body
        assert isinstance(body[key], list)


def test_analyze_image_populates_graph_node_id_on_every_detected_item(client):
    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("graph_node_check.png", _png_bytes_with_object(), "image/png")},
        data={"source_type": "inspection_photo"},
    )
    image_id = upload.json()["id"]

    response = client.post(f"/api/v1/vision/images/{image_id}/analyze")

    body = response.json()
    assert len(body["detections"]) >= 1
    for detection in body["detections"]:
        assert detection["graph_node_id"] is not None


def test_analyze_missing_image_returns_404(client):
    response = client.post("/api/v1/vision/images/does-not-exist/analyze")

    assert response.status_code == 404


def test_multimodal_query_returns_evidence_list_shape(client):
    response = client.post("/api/v1/vision/query", json={"query": "What is the weather today?", "top_k": 5})

    assert response.status_code == 200
    assert "evidence" in response.json()
    assert isinstance(response.json()["evidence"], list)


def test_multimodal_query_includes_visual_evidence_for_analyzed_equipment(client):
    client.post(
        "/api/v1/documents/upload",
        files={"file": ("vision_query_doc.txt", b"Pump P-9500 is scheduled for inspection.", "text/plain")},
    )

    upload = client.post(
        "/api/v1/vision/images",
        files={"file": ("p9500.png", _png_bytes_with_object(), "image/png")},
        data={"source_type": "equipment_photo", "equipment_hint": "P-9500"},
    )
    image_id = upload.json()["id"]
    client.post(f"/api/v1/vision/images/{image_id}/analyze")

    response = client.post("/api/v1/vision/query", json={"query": "What is the condition of P-9500?", "top_k": 8})

    assert response.status_code == 200
    sources = {e["source"] for e in response.json()["evidence"]}
    assert any(source.startswith("vision:") for source in sources)
