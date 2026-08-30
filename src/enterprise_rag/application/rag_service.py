"""Application orchestration boundary for the existing RAG components."""
from dataclasses import dataclass
from typing import Any
from enterprise_rag.routing import understand_query
from enterprise_rag.observability import RuntimeTracer
from enterprise_rag.config import get_settings

@dataclass(frozen=True, slots=True)
class QueryOutcome:
    answer: str | None
    sources: tuple[Any, ...]
    request_id: str
    clarification_question: str | None = None

class RAGService:
    def __init__(self, retriever: Any, generator: Any, reranker: Any | None = None, tracer: RuntimeTracer | None = None):
        self.retriever, self.generator, self.reranker, self.tracer = retriever, generator, reranker, tracer or RuntimeTracer(enabled=False)

    def query(self, text: str) -> QueryOutcome:
        trace = self.tracer.start(text)
        try:
            with self.tracer.stage(trace, "routing"):
                routed = understand_query(text)
            if routed.clarification_required:
                self.tracer.finish(trace, source_ids=())
                return QueryOutcome(None, (), trace.request_id, routed.clarification_question)
            with self.tracer.stage(trace, "retrieval"):
                results = self.retriever.retrieve(routed.retrieval_query, metadata_filter=routed.constraints)
            if self.reranker is not None:
                with self.tracer.stage(trace, "reranking"):
                    results = self.reranker.rerank(routed.retrieval_query, results)
            with self.tracer.stage(trace, "generation"):
                generated, context = self.generator.generate(routed.retrieval_query, results)
            sources = tuple(item for item in context if item.chunk_id in generated.source_ids)
            self.tracer.finish(trace, model_name=getattr(generated, "model_name", None), usage=getattr(generated, "usage", None), source_ids=generated.source_ids)
            return QueryOutcome(generated.answer, sources, trace.request_id)
        except Exception as exc:
            self.tracer.fail(trace, "query", exc)
            raise

def build_default_service() -> RAGService:
    settings = get_settings()
    from enterprise_rag.embeddings import SentenceTransformerEmbeddingService
    from enterprise_rag.stores import ChromaVectorStore
    from enterprise_rag.retrieval import SemanticRetriever, BM25Retriever, HybridRetriever
    from enterprise_rag.generation import GeminiService, GroundedGenerator, ContextBuilder
    embedding = SentenceTransformerEmbeddingService(settings.embedding_model_name, settings.embedding_batch_size)
    store = ChromaVectorStore(settings.chroma_db_path, settings.chroma_collection_name, settings.embedding_model_name, embedding.dimension)
    semantic = SemanticRetriever(embedding, store, settings.semantic_top_k)
    artifact = next(settings.chunks_dir.glob("*.chunks.json"), None)
    if artifact is None: raise RuntimeError("No chunk artifact is available")
    hybrid = HybridRetriever(semantic, BM25Retriever.from_chunk_artifact(artifact, settings.bm25_top_k), settings.hybrid_top_k, settings.hybrid_candidate_depth, settings.rrf_k)
    reranker = None
    if settings.reranker_enabled:
        from enterprise_rag.reranking import CrossEncoderReranker
        reranker = CrossEncoderReranker(settings.reranker_model_name, settings.reranker_candidate_depth, settings.reranker_top_k, enabled=True)
    generator = GroundedGenerator(GeminiService(settings.gemini_api_key, settings.gemini_model_name), ContextBuilder(settings.context_max_chars))
    tracer = RuntimeTracer(settings.observability_enabled, settings.trace_path if settings.trace_persistence_enabled else None)
    return RAGService(hybrid, generator, reranker, tracer)

class LazyRAGService:
    def __init__(self): self._service = None
    def query(self, text):
        if self._service is None: self._service = build_default_service()
        return self._service.query(text)
