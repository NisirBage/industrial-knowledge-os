"""Vision Ingestion Service (M7 Component 1): the image analog of
IngestionService — validates and stores an uploaded image as a first-class
VisionDocument. No assumption about image source: source_type is supplied by
the caller (upload form), since classifying "is this a P&ID or a photo" from
pixels alone is out of scope for a rule-based/classical-CV baseline (a real
vision-language model could do this later — see docs/architecture/vision.md).

Storage is local disk (matching the ML Registry's ML_MODELS_DIR precedent —
no MinIO yet, see docs/roadmap/README.md), tracked via VisionMetadataRepository.
"""

import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.vision.ingestion.models import VisionDocument, VisionSourceType
from app.vision.ingestion.repository import VisionMetadataRepository


class UnsupportedImageError(ValueError):
    pass


class VisionIngestionService:
    def __init__(self, repository: VisionMetadataRepository, storage_dir: Path):
        self._repository = repository
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    async def ingest(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str,
        source_type: VisionSourceType,
        equipment_hint: str | None = None,
    ) -> VisionDocument:
        try:
            image = Image.open(BytesIO(image_bytes))
            image.verify()
            image = Image.open(BytesIO(image_bytes))  # verify() consumes the file handle
            width, height = image.size
        except (UnidentifiedImageError, OSError) as exc:
            raise UnsupportedImageError(f"Could not decode image {filename!r}: {exc}") from exc

        document_id = str(uuid.uuid4())
        suffix = Path(filename).suffix or ".png"
        storage_path = self._storage_dir / f"{document_id}{suffix}"
        storage_path.write_bytes(image_bytes)

        document = VisionDocument(
            id=document_id,
            filename=filename,
            source_type=source_type,
            content_type=content_type,
            width=width,
            height=height,
            storage_path=str(storage_path),
            size_bytes=len(image_bytes),
            equipment_hint=equipment_hint,
        )
        await self._repository.save(document)
        return document
