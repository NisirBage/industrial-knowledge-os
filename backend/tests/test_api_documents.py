def test_upload_txt_document_returns_chunk_and_entity_counts(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("audit_test_doc.txt", b"Pump P-9001 failed due to Valve V-9002 blockage.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "audit_test_doc.txt"
    assert body["num_chunks"] >= 1
    assert "document_id" in body


def test_uploaded_document_appears_in_list(client):
    upload = client.post(
        "/api/v1/documents/upload",
        files={"file": ("list_check.txt", b"Motor M-100 is operational.", "text/plain")},
    )
    document_id = upload.json()["document_id"]

    listing = client.get("/api/v1/documents")

    assert listing.status_code == 200
    ids = {d["id"] for d in listing.json()}
    assert document_id in ids


def test_unsupported_file_extension_returns_422_not_500(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", b"not a real document", "application/octet-stream")},
    )

    assert response.status_code == 422
    assert ".exe" in response.json()["detail"]


def test_oversized_upload_returns_413(client):
    oversized = b"x" * (26 * 1024 * 1024)  # over the 25MB default limit

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("huge.txt", oversized, "text/plain")},
    )

    assert response.status_code == 413


def test_upload_missing_file_field_returns_422(client):
    response = client.post("/api/v1/documents/upload")

    assert response.status_code == 422


def test_documents_upload_rejects_get(client):
    response = client.get("/api/v1/documents/upload")

    assert response.status_code == 405
