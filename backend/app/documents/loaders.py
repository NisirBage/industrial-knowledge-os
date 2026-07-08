"""Document loaders: convert a raw file on disk into structured page text.

MVP supports PDF and plain text. Future loaders (DOCX, XLSX, scanned images via
OCR, P&ID drawings) plug in here without touching the rest of the pipeline —
each loader only needs to produce a list[PageContent].
"""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class PageContent:
    page_number: int
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    filename: str
    content_type: str
    pages: list[PageContent]


def load_pdf(path: Path) -> LoadedDocument:
    pages: list[PageContent] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=i + 1, text=text))
    return LoadedDocument(filename=path.name, content_type="application/pdf", pages=pages)


def load_text(path: Path) -> LoadedDocument:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return LoadedDocument(filename=path.name, content_type="text/plain", pages=[PageContent(page_number=1, text=text)])


_LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_text,
    ".md": load_text,
}


def supported_extensions() -> set[str]:
    return set(_LOADERS)


def load_document(path: Path) -> LoadedDocument:
    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported document type: {suffix}")
    return loader(path)
