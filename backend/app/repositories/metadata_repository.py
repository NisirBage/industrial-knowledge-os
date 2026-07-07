"""Document metadata store.

MVP substitute for Postgres: SQLite via SQLAlchemy's async engine, behind the
same repository interface, so swapping the URL to a real Postgres DSN in
production requires no code change (see METADATA_DB_URL in config).
"""

import datetime as dt
import uuid

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String)
    content_type: Mapped[str] = mapped_column(String)
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    num_entities: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class MetadataRepository:
    def __init__(self, db_url: str):
        self._engine = create_async_engine(db_url)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init_models(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_document(self, filename: str, content_type: str) -> str:
        document_id = str(uuid.uuid4())
        async with self._session_factory() as session:
            session.add(DocumentRecord(id=document_id, filename=filename, content_type=content_type))
            await session.commit()
        return document_id

    async def update_counts(self, document_id: str, num_chunks: int, num_entities: int) -> None:
        async with self._session_factory() as session:
            record = await session.get(DocumentRecord, document_id)
            if record is not None:
                record.num_chunks = num_chunks
                record.num_entities = num_entities
                await session.commit()

    async def list_documents(self) -> list[DocumentRecord]:
        from sqlalchemy import select

        async with self._session_factory() as session:
            result = await session.execute(select(DocumentRecord))
            return list(result.scalars().all())
