"""Vision document metadata store — the image analog of
app/repositories/metadata_repository.py. Own database, per the
independent-subsystem convention every milestone since M4 has followed."""

import datetime as dt

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.vision.ingestion.models import VisionDocument, VisionSourceType


class Base(DeclarativeBase):
    pass


class VisionDocumentRecord(Base):
    __tablename__ = "vision_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    source_type: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    storage_path: Mapped[str] = mapped_column(String)
    size_bytes: Mapped[int] = mapped_column(Integer)
    equipment_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


def _record_to_document(record: VisionDocumentRecord) -> VisionDocument:
    return VisionDocument(
        id=record.id,
        filename=record.filename,
        source_type=VisionSourceType(record.source_type),
        content_type=record.content_type,
        width=record.width,
        height=record.height,
        storage_path=record.storage_path,
        size_bytes=record.size_bytes,
        equipment_hint=record.equipment_hint,
    )


class VisionMetadataRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def save(self, document: VisionDocument) -> None:
        async with self._session_factory() as session:
            session.add(
                VisionDocumentRecord(
                    id=document.id,
                    filename=document.filename,
                    source_type=document.source_type.value,
                    content_type=document.content_type,
                    width=document.width,
                    height=document.height,
                    storage_path=document.storage_path,
                    size_bytes=document.size_bytes,
                    equipment_hint=document.equipment_hint,
                )
            )
            await session.commit()

    async def get(self, document_id: str) -> VisionDocument | None:
        async with self._session_factory() as session:
            record = await session.get(VisionDocumentRecord, document_id)
            return _record_to_document(record) if record else None

    async def list(self) -> list[VisionDocument]:
        from sqlalchemy import select

        async with self._session_factory() as session:
            result = await session.execute(select(VisionDocumentRecord))
            return [_record_to_document(r) for r in result.scalars().all()]
