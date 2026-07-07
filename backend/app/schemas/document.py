from pydantic import BaseModel


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    num_entities: int
    num_relations: int


class DocumentSummary(BaseModel):
    id: str
    filename: str
    content_type: str
    num_chunks: int
    num_entities: int
