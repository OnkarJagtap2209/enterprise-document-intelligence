"""Semantic retrieval over the Phase 4 vector index."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from enterprise_rag.routing import QueryConstraints, matches_constraints


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
    provenance: tuple[dict[str, Any], ...] = ()


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
        self, query: str, top_k: int | None = None, metadata_filter: QueryConstraints | None = None
    ) -> tuple[SemanticRetrievalResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise SemanticRetrievalError("query must be a non-empty string")
        selected_top_k = self.default_top_k if top_k is None else top_k
        _validate_top_k(selected_top_k)

        query_embedding = self.embedding_service.embed_query(query.strip())
        if metadata_filter is None:
            matches = self.vector_store.semantic_query(query_embedding, selected_top_k)
        else:
            query_depth = selected_top_k
            if metadata_filter.year is not None and hasattr(self.vector_store, "count"):
                query_depth = max(selected_top_k, int(self.vector_store.count()))
            matches = self.vector_store.semantic_query(query_embedding, query_depth)
        results = tuple(
            SemanticRetrievalResult(
                chunk_id=match.chunk_id,
                document_id=match.metadata["document_id"],
                content=match.document,
                metadata=match.metadata,
                distance=match.distance,
                rank=rank,
                provenance=_parse_provenance(match.metadata.get("provenance_json")),
            )
            for rank, match in enumerate(matches, start=1)
        )
        if metadata_filter is not None:
            results = tuple(result for result in results if matches_constraints(result.metadata, metadata_filter))
        return results[:selected_top_k]


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise SemanticRetrievalError("top_k must be greater than zero")


def _parse_provenance(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, str):
        return ()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(item for item in parsed if isinstance(item, dict))
