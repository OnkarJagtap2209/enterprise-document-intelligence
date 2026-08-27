"""Index persisted Phase 3 chunks as one vector record per chunk."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from enterprise_rag.stores.vector_store import chunk_metadata_to_chroma


class IndexingError(ValueError):
    """Raised when a chunk artifact cannot be indexed safely."""


@dataclass(frozen=True, slots=True)
class IndexingResult:
    artifact_path: Path
    chunk_count: int
    embeddings_created: int
    embedding_dimension: int
    collection_name: str
    collection_record_count: int


class ChunkIndexer:
    """Batch-embed a chunk artifact and upsert it into ChromaDB."""

    def __init__(self, embedding_service: Any, vector_store: Any, batch_size: int) -> None:
        if batch_size <= 0:
            raise IndexingError("batch_size must be greater than zero")
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index(self, artifact_path: str | Path) -> IndexingResult:
        path, chunks = load_chunk_artifact(artifact_path)
        embedded_count = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            documents = [chunk["content"] for chunk in batch]
            embeddings = self.embedding_service.embed_documents(documents)
            ids = [chunk["metadata"]["chunk_id"] for chunk in batch]
            metadatas = [
                chunk_metadata_to_chroma(
                    chunk["metadata"], provenance=chunk.get("provenance")
                )
                for chunk in batch
            ]
            self.vector_store.upsert(ids, embeddings, documents, metadatas)
            embedded_count += len(embeddings)

        return IndexingResult(
            artifact_path=path,
            chunk_count=len(chunks),
            embeddings_created=embedded_count,
            embedding_dimension=self.embedding_service.dimension,
            collection_name=self.vector_store.collection_name,
            collection_record_count=self.vector_store.count(),
        )


def load_chunk_artifact(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise IndexingError(f"Chunk artifact does not exist: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndexingError(f"Could not load chunk artifact: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("artifact_schema") != "enterprise_rag.chunks.v1":
        raise IndexingError("Unsupported or missing chunk artifact schema")
    chunks = artifact.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise IndexingError("Chunk artifact contains no chunks")

    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("content"), str) or not chunk["content"].strip():
            raise IndexingError(f"Chunk {index} has no content")
        metadata = chunk.get("metadata")
        chunk_id = metadata.get("chunk_id") if isinstance(metadata, dict) else None
        if not isinstance(chunk_id, str) or not chunk_id:
            raise IndexingError(f"Chunk {index} has no deterministic chunk_id")
        if chunk_id in seen:
            raise IndexingError(f"Duplicate chunk_id in artifact: {chunk_id}")
        seen.add(chunk_id)
    return artifact_path.resolve(), chunks
