from fastapi import FastAPI, HTTPException
from enterprise_rag.api.schemas import QueryRequest, QueryResponse, SourceResponse, ClarificationResponse
from enterprise_rag.application import LazyRAGService

def create_app(rag_service=None) -> FastAPI:
    app = FastAPI(title="Enterprise Document Intelligence")
    app.state.rag_service = rag_service
    @app.get("/health")
    def health(): return {"status": "ok"}
    @app.post("/query", response_model=QueryResponse | ClarificationResponse)
    def query(request: QueryRequest):
        if app.state.rag_service is None: raise HTTPException(503, "RAG service is not configured")
        try: outcome = app.state.rag_service.query(request.query)
        except Exception as exc: raise HTTPException(500, "Unable to process the query.") from exc
        if outcome.clarification_question: return ClarificationResponse(clarification_question=outcome.clarification_question, request_id=outcome.request_id)
        sources = [SourceResponse(chunk_id=item.chunk_id, document_id=item.document_id, source_filename=item.metadata.get("source_filename"), page_start=item.metadata.get("page_start"), page_end=item.metadata.get("page_end")) for item in outcome.sources]
        return QueryResponse(answer=outcome.answer or "", sources=sources, request_id=outcome.request_id)
    return app

app = create_app(LazyRAGService())
