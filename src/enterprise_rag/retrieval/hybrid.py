"""Rank-only hybrid retrieval using semantic, BM25, and RRF components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from enterprise_rag.retrieval.rrf import reciprocal_rank_fusion


class HybridRetrievalError(ValueError):
    """Raised when hybrid retrieval configuration or requests are invalid."""


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    provenance: tuple[dict[str, Any], ...]
    rrf_score: float
    retriever_ranks: dict[str, int]
    rank: int


class HybridRetriever:
    """Orchestrate independent retrievers and fuse their one-based ranks."""

    def __init__(
        self,
        semantic_retriever: Any,
        bm25_retriever: Any,
        default_top_k: int = 5,
        candidate_depth: int = 10,
        rrf_k: int = 60,
    ) -> None:
        _validate_positive(default_top_k, "top_k")
        _validate_positive(candidate_depth, "candidate_depth")
        _validate_positive(rrf_k, "rrf_k")
        self.semantic_retriever = semantic_retriever
        self.bm25_retriever = bm25_retriever
        self.default_top_k = default_top_k
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> tuple[HybridRetrievalResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise HybridRetrievalError("query must be a non-empty string")
        selected = self.default_top_k if top_k is None else top_k
        _validate_positive(selected, "top_k")
        semantic = tuple(self.semantic_retriever.retrieve(query, top_k=self.candidate_depth))
        lexical = tuple(self.bm25_retriever.retrieve(query, top_k=self.candidate_depth))
        rankings = {
            "semantic": tuple((item.chunk_id, item.rank) for item in semantic),
            "bm25": tuple((item.chunk_id, item.rank) for item in lexical),
        }
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k, top_k=selected)
        by_id = {item.chunk_id: item for item in (*semantic, *lexical)}
        return tuple(
            HybridRetrievalResult(
                chunk_id=item.chunk_id,
                document_id=source.document_id,
                content=source.content,
                metadata=dict(source.metadata),
                provenance=tuple(getattr(source, "provenance", ())),
                rrf_score=item.rrf_score,
                retriever_ranks=dict(item.retriever_ranks),
                rank=rank,
            )
            for rank, item in enumerate(fused, start=1)
            if (source := by_id.get(item.chunk_id)) is not None
        )


def _validate_positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise HybridRetrievalError(f"{name} must be greater than zero")
