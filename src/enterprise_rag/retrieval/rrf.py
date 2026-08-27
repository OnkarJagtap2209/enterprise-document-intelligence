"""Independently testable Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


class ReciprocalRankFusionError(ValueError):
    """Raised when ranked inputs or RRF configuration are invalid."""


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusionResult:
    chunk_id: str
    rrf_score: float
    retriever_ranks: dict[str, int]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[tuple[str, int]]],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> tuple[ReciprocalRankFusionResult, ...]:
    """Fuse one-based ranks using sum(1 / (rrf_k + rank))."""
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
        raise ReciprocalRankFusionError("rrf_k must be greater than zero")
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
    ):
        raise ReciprocalRankFusionError("top_k must be greater than zero")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    for retriever_name, ranked_items in rankings.items():
        if not retriever_name:
            raise ReciprocalRankFusionError("retriever names must not be empty")
        seen: set[str] = set()
        for chunk_id, rank in ranked_items:
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ReciprocalRankFusionError("chunk IDs must not be empty")
            if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
                raise ReciprocalRankFusionError("ranks must be greater than zero")
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + rank)
            ranks.setdefault(chunk_id, {})[retriever_name] = rank

    fused = [
        ReciprocalRankFusionResult(
            chunk_id=chunk_id,
            rrf_score=score,
            retriever_ranks=dict(ranks[chunk_id]),
        )
        for chunk_id, score in scores.items()
    ]
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            min(item.retriever_ranks.values()),
            item.chunk_id,
        )
    )
    return tuple(fused if top_k is None else fused[:top_k])
