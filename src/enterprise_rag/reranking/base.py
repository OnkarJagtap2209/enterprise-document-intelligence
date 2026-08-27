"""Common reranking result and interface types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


class RerankingError(ValueError):
    """Raised when reranking input or configuration is invalid."""


@dataclass(frozen=True, slots=True)
class RerankingResult:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    provenance: tuple[dict[str, Any], ...]
    reranker_score: float
    rank: int


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self, query: str, candidates: Sequence[Any], top_k: int | None = None
    ) -> tuple[Any, ...]:
        """Return candidates in reranked order."""
