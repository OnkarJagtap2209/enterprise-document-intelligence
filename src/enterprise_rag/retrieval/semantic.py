"""Semantic retrieval over the Phase 4 vector index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SemanticRetrievalError(ValueError):
    """Raised when a semantic retrieval request is invalid."""


@dataclass(frozen=True, slots=True)
class SemanticRetrievalResult:
    """One retrieved chunk; distance is raw cosine distance (lower is nearer)."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    distance: float
    rank: int


class SemanticRetriever:
    """Embed a query once and preserve ChromaDB's nearest-neighbor ordering."""

    def __init__(
        self, embedding_service: Any, vector_store: Any, default_top_k: int = 5
    ) -> None:
        _validate_top_k(default_top_k)
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(
        self, query: str, top_k: int | None = None
    ) -> tuple[SemanticRetrievalResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise SemanticRetrievalError("query must be a non-empty string")
        selected_top_k = self.default_top_k if top_k is None else top_k
        _validate_top_k(selected_top_k)

        query_embedding = self.embedding_service.embed_query(query.strip())
        matches = self.vector_store.semantic_query(query_embedding, selected_top_k)
        return tuple(
            SemanticRetrievalResult(
                chunk_id=match.chunk_id,
                document_id=match.metadata["document_id"],
                content=match.document,
                metadata=match.metadata,
                distance=match.distance,
                rank=rank,
            )
            for rank, match in enumerate(matches, start=1)
        )


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise SemanticRetrievalError("top_k must be greater than zero")
