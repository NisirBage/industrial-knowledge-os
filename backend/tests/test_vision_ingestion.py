from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.vision.ingestion.models import VisionSourceType
from app.vision.ingestion.repository import VisionMetadataRepository
from app.vision.ingestion.service import UnsupportedImageError, VisionIngestionService


def _png_bytes(width: int = 64, height: int = 48, color=(255, 0, 0)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


async def _service(tmp_path) -> VisionIngestionService:
    repository = VisionMetadataRepository(f"sqlite+aiosqlite:///{tmp_path}/vision.db")
    await repository.init_models()
    return VisionIngestionService(repository, tmp_path / "images")


async def test_ingest_valid_png_returns_document_with_correct_dimensions(tmp_path):
    service = await _service(tmp_path)

    document = await service.ingest(
        _png_bytes(64, 48), "gauge.png", "image/png", VisionSourceType.EQUIPMENT_PHOTO
    )

    assert document.width == 64
    assert document.height == 48
    assert document.source_type == VisionSourceType.EQUIPMENT_PHOTO
    assert Path(document.storage_path).exists()


async def test_ingest_stores_equipment_hint(tmp_path):
    service = await _service(tmp_path)

    document = await service.ingest(
        _png_bytes(), "p101.png", "image/png", VisionSourceType.INSPECTION_PHOTO, equipment_hint="P-101"
    )

    assert document.equipment_hint == "P-101"


async def test_ingest_rejects_non_image_bytes(tmp_path):
    service = await _service(tmp_path)

    with pytest.raises(UnsupportedImageError):
        await service.ingest(b"this is not an image", "fake.png", "image/png", VisionSourceType.EQUIPMENT_PHOTO)


async def test_ingested_document_is_retrievable_and_listed(tmp_path):
    service = await _service(tmp_path)
    document = await service.ingest(_png_bytes(), "diagram.png", "image/png", VisionSourceType.PID_DIAGRAM)

    fetched = await service._repository.get(document.id)
    listed = await service._repository.list()

    assert fetched is not None
    assert fetched.id == document.id
    assert any(d.id == document.id for d in listed)


async def test_ingest_supports_every_documented_source_type(tmp_path):
    service = await _service(tmp_path)

    for source_type in VisionSourceType:
        document = await service.ingest(_png_bytes(), f"{source_type.value}.png", "image/png", source_type)
        assert document.source_type == source_type
