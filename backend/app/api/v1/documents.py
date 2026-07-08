import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.documents.loaders import supported_extensions
from app.schemas.document import DocumentSummary, DocumentUploadResponse

router = APIRouter(tags=["documents"])

_READ_CHUNK_BYTES = 1024 * 1024


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile) -> DocumentUploadResponse:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in supported_extensions():
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported document type {suffix!r}. Supported: {sorted(supported_extensions())}",
        )

    max_bytes = request.app.state.settings.max_upload_size_bytes
    size_so_far = 0

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        while chunk := await file.read(_READ_CHUNK_BYTES):
            size_so_far += len(chunk)
            if size_so_far > max_bytes:
                tmp.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {request.app.state.settings.max_upload_size_mb}MB upload limit.",
                )
            tmp.write(chunk)

    ingestion_service = request.app.state.ingestion_service
    try:
        return await ingestion_service.ingest(
            tmp_path,
            original_filename=file.filename or tmp_path.name,
            content_type=file.content_type or "application/octet-stream",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(request: Request) -> list[DocumentSummary]:
    metadata_repository = request.app.state.metadata_repository
    records = await metadata_repository.list_documents()
    return [
        DocumentSummary(
            id=r.id,
            filename=r.filename,
            content_type=r.content_type,
            num_chunks=r.num_chunks,
            num_entities=r.num_entities,
        )
        for r in records
    ]
