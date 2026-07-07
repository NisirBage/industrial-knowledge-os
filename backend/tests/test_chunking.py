from app.documents.chunking import chunk_pages
from app.documents.loaders import LoadedDocument, PageContent


def test_chunk_pages_splits_long_page_into_multiple_chunks():
    long_paragraph_a = "Pump P-101 inspection notes. " * 40
    long_paragraph_b = "Valve V-2201A maintenance history. " * 40
    document = LoadedDocument(
        filename="manual.pdf",
        content_type="application/pdf",
        pages=[PageContent(page_number=1, text=f"{long_paragraph_a}\n\n{long_paragraph_b}")],
    )

    chunks = chunk_pages(document, document_id="doc-1", chunk_size=500, overlap=50)

    assert len(chunks) > 1
    assert all(c.document_id == "doc-1" for c in chunks)
    assert all(c.page_number == 1 for c in chunks)


def test_chunk_pages_empty_page_produces_no_chunks():
    document = LoadedDocument(filename="empty.pdf", content_type="application/pdf", pages=[PageContent(page_number=1, text="")])

    chunks = chunk_pages(document, document_id="doc-2")

    assert chunks == []
