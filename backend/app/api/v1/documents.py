import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile

from app.schemas.document import DocumentSummary, DocumentUploadResponse

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(request: Request, file: UploadFile) -> DocumentUploadResponse:
    ingestion_service = request.app.state.ingestion_service
    suffix = Path(file.filename or "upload").suffix

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

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
