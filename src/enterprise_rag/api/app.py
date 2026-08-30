from fastapi import FastAPI, HTTPException, UploadFile, File
import logging
from enterprise_rag.api.schemas import QueryRequest, QueryResponse, SourceResponse, ClarificationResponse, DocumentUploadResponse
from enterprise_rag.application import LazyRAGService, DocumentService, DocumentProcessingError

logger = logging.getLogger(__name__)

def create_app(rag_service=None, document_service=None) -> FastAPI:
    app = FastAPI(title="Enterprise Document Intelligence")
    app.state.rag_service = rag_service
    app.state.document_service = document_service
    @app.get("/health")
    def health(): return {"status": "ok"}
    @app.post("/query", response_model=QueryResponse | ClarificationResponse)
    def query(request: QueryRequest):
        if app.state.rag_service is None: raise HTTPException(503, "RAG service is not configured")
        try: outcome = app.state.rag_service.query(request.query)
        except Exception as exc:
            logger.exception("Query processing failed")
            raise HTTPException(500, "Unable to process the query.") from exc
        if outcome.clarification_question: return ClarificationResponse(clarification_question=outcome.clarification_question, request_id=outcome.request_id)
        sources = [SourceResponse(chunk_id=item.chunk_id, document_id=item.document_id, source_filename=item.metadata.get("source_filename"), page_start=item.metadata.get("page_start"), page_end=item.metadata.get("page_end")) for item in outcome.sources]
        return QueryResponse(answer=outcome.answer or "", sources=sources, request_id=outcome.request_id)
    @app.post("/documents", response_model=DocumentUploadResponse)
    async def upload_document(file: UploadFile = File(...)):
        if app.state.document_service is None:
            app.state.document_service = DocumentService()
        try:
            content = await file.read()
            result = app.state.document_service.process(
                file.filename or "", content,
                on_indexed=getattr(app.state.rag_service, "refresh", None),
            )
        except DocumentProcessingError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, "Unable to process the uploaded document.") from exc
        return DocumentUploadResponse(
            document_id=result.document_id,
            source_filename=result.source_filename,
            chunk_count=result.chunk_count,
            indexed_count=result.indexed_count,
            status=result.status,
        )
    return app

app = create_app(LazyRAGService())
