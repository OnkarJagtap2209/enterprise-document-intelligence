from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)

class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    source_filename: str | None = None
    page_start: int | None = None
    page_end: int | None = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    request_id: str

class ClarificationResponse(BaseModel):
    clarification_required: bool = True
    clarification_question: str
    request_id: str
