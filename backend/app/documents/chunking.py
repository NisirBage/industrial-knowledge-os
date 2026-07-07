"""Chunk generation over loaded document pages.

Splits page text into overlapping, roughly fixed-size chunks along paragraph
boundaries where possible, so downstream embedding + retrieval operate on
coherent units instead of arbitrary character windows.
"""

import uuid
from dataclasses import dataclass, field

from app.documents.loaders import LoadedDocument


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def chunk_pages(
    document: LoadedDocument,
    document_id: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        paragraphs = _split_paragraphs(page.text) or [page.text]
        buffer = ""

        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue

            if buffer:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        text=buffer,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        metadata={"filename": document.filename},
                    )
                )
                chunk_index += 1
                buffer = buffer[-overlap:] + "\n\n" + paragraph if overlap else paragraph
            else:
                buffer = paragraph

        if buffer:
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_id=document_id,
                    text=buffer,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    metadata={"filename": document.filename},
                )
            )
            chunk_index += 1

    return chunks
