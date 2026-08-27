"""Small ChromaDB boundary for persistent chunk vectors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class VectorStoreError(ValueError):
    """Raised when a collection or vector batch is invalid."""


@dataclass(frozen=True, slots=True)
class CollectionStats:
    name: str
    record_count: int
    metadata: dict[str, Any]
    configuration: dict[str, Any]


class ChromaVectorStore:
    """Create and write one Chroma collection with explicit embeddings."""

    SCHEMA_VERSION = "enterprise_rag.vector.v1"

    def __init__(
        self,
        db_path: str | Path,
        collection_name: str,
        embedding_model: str,
        embedding_dimension: int,
        client: Any | None = None,
    ) -> None:
        if not collection_name.strip():
            raise VectorStoreError("collection_name must not be empty")
        if not embedding_model.strip():
            raise VectorStoreError("embedding_model must not be empty")
        if embedding_dimension <= 0:
            raise VectorStoreError("embedding_dimension must be greater than zero")
        if client is None:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(Path(db_path).expanduser()))
            except Exception as exc:
                raise VectorStoreError(f"Could not initialize ChromaDB: {exc}") from exc

        expected_metadata = {
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "schema_version": self.SCHEMA_VERSION,
            "chunk_schema": "enterprise_rag.chunks.v1",
        }
        try:
            self.collection = client.get_or_create_collection(
                name=collection_name,
                metadata=expected_metadata,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not create or open collection: {exc}") from exc
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self._validate_collection(expected_metadata)

    def _validate_collection(self, expected: Mapping[str, Any]) -> None:
        actual = self.collection.metadata or {}
        mismatches = [key for key, value in expected.items() if actual.get(key) != value]
        if mismatches:
            raise VectorStoreError(
                "Collection configuration mismatch for: " + ", ".join(mismatches)
            )
        space = (self.collection.configuration or {}).get("hnsw", {}).get("space")
        if space != "cosine":
            raise VectorStoreError("Collection must use cosine distance")

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        """Insert or replace records; stable IDs make repeated runs idempotent."""
        count = len(ids)
        if count == 0:
            raise VectorStoreError("At least one vector record is required")
        if len(set(ids)) != count:
            raise VectorStoreError("Vector record IDs must be unique within a batch")
        if not (len(embeddings) == len(documents) == len(metadatas) == count):
            raise VectorStoreError("Vector record fields must have equal lengths")
        if any(len(vector) != self.embedding_dimension for vector in embeddings):
            raise VectorStoreError("Vector has an unexpected embedding dimension")
        try:
            self.collection.upsert(
                ids=list(ids),
                embeddings=[list(vector) for vector in embeddings],
                documents=list(documents),
                metadatas=[dict(metadata) for metadata in metadatas],
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not upsert vector records: {exc}") from exc

    def count(self) -> int:
        return int(self.collection.count())

    def stats(self) -> CollectionStats:
        return CollectionStats(
            name=self.collection.name,
            record_count=self.count(),
            metadata=dict(self.collection.metadata or {}),
            configuration=dict(self.collection.configuration or {}),
        )

    def peek(self, limit: int = 5) -> Mapping[str, Any]:
        if limit <= 0:
            raise VectorStoreError("peek limit must be greater than zero")
        return self.collection.peek(limit=limit)


def chunk_metadata_to_chroma(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Phase 3 metadata to deterministic Chroma-compatible scalars."""
    required = (
        "chunk_id",
        "document_id",
        "source_filename",
        "pages",
        "section_context",
        "content_type",
        "chunk_index",
        "source_refs",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise VectorStoreError("Chunk metadata is missing: " + ", ".join(missing))

    pages = _list_value(metadata["pages"], "pages")
    sections = _list_value(metadata["section_context"], "section_context")
    refs = _list_value(metadata["source_refs"], "source_refs")
    labels = _list_value(metadata.get("structural_labels", []), "structural_labels")
    converted: dict[str, Any] = {
        "chunk_id": str(metadata["chunk_id"]),
        "document_id": str(metadata["document_id"]),
        "source_filename": str(metadata["source_filename"]),
        "content_type": str(metadata["content_type"]),
        "chunk_index": int(metadata["chunk_index"]),
        "pages_json": _json_scalar(pages),
        "section_context_json": _json_scalar(sections),
        "section_path": " > ".join(str(value) for value in sections),
        "source_refs_json": _json_scalar(refs),
        "structural_labels_json": _json_scalar(labels),
    }
    if pages:
        converted["page_start"] = int(min(pages))
        converted["page_end"] = int(max(pages))
    return converted


def _list_value(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise VectorStoreError(f"{field_name} must be a list")
    return list(value)


def _json_scalar(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
