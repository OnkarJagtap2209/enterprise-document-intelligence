"""Application workflow for safely processing an uploaded PDF."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import logging
from typing import Any, Callable

from enterprise_rag.config import Settings, get_settings
from enterprise_rag.ingestion import ChunkingPipeline, IngestionPipeline, StructureAwareChunker
from enterprise_rag.embeddings import SentenceTransformerEmbeddingService
from enterprise_rag.stores import ChromaVectorStore, ChunkIndexer

logger = logging.getLogger(__name__)


class DocumentProcessingError(RuntimeError):
    """Raised when an uploaded document cannot be processed."""


@dataclass(frozen=True, slots=True)
class DocumentProcessingResult:
    document_id: str
    source_filename: str
    chunk_count: int
    indexed_count: int
    status: str = "ready"


class DocumentService:
    """Reuse the canonical ingestion, chunking, and indexing components."""

    def __init__(self, settings: Settings | None = None, components: tuple[Any, Any, Any] | None = None) -> None:
        self.settings = settings or get_settings()
        self._components = components

    def process(self, filename: str, content: bytes, on_indexed: Callable[[], None] | None = None) -> DocumentProcessingResult:
        safe_name = _safe_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            raise DocumentProcessingError("Only PDF uploads are supported.")
        if not content:
            raise DocumentProcessingError("The uploaded PDF is empty.")
        self.settings.document_dir.mkdir(parents=True, exist_ok=True)
        digest = sha256(content).hexdigest()
        path = self.settings.document_dir / safe_name
        try:
            if path.exists() and path.read_bytes() != content:
                path = self.settings.document_dir / f"{Path(safe_name).stem}.{digest[:12]}.pdf"
        except OSError as exc:
            raise DocumentProcessingError("Could not access the uploaded PDF.") from exc
        try:
            path.write_bytes(content)
        except OSError as exc:
            raise DocumentProcessingError("Could not store the uploaded PDF.") from exc
        try:
            ingestion = IngestionPipeline(self.settings.extracted_dir).ingest(path)
            chunking = ChunkingPipeline(
                self.settings.chunks_dir,
                StructureAwareChunker(self.settings.chunk_max_chars, self.settings.chunk_overlap_chars),
            ).run(ingestion.output_path)
            embedding, store, _ = self._build_components()
            indexed = ChunkIndexer(embedding, store, self.settings.embedding_batch_size).index(chunking.output_path)
            if on_indexed is not None:
                on_indexed()
        except Exception as exc:
            logger.exception("Document processing failed at upload pipeline for %s", safe_name)
            raise DocumentProcessingError("The PDF could not be processed.") from exc
        return DocumentProcessingResult(ingestion.document_id, safe_name, len(chunking.chunks), indexed.chunk_count)

    def _build_components(self) -> tuple[Any, Any, Any]:
        if self._components is not None:
            return self._components
        embedding = SentenceTransformerEmbeddingService(self.settings.embedding_model_name, self.settings.embedding_batch_size)
        store = ChromaVectorStore(self.settings.chroma_db_path, self.settings.chroma_collection_name, self.settings.embedding_model_name, embedding.dimension)
        self._components = (embedding, store, None)
        return self._components


def _safe_filename(filename: str) -> str:
    if not isinstance(filename, str) or not filename.strip():
        raise DocumentProcessingError("A filename is required.")
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name or name in {".", ".."}:
        raise DocumentProcessingError("The filename is unsafe.")
    return name
