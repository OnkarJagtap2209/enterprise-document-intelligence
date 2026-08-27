"""Optional sentence-transformers cross-encoder reranking."""

from __future__ import annotations

from math import isfinite
from typing import Any, Sequence

from enterprise_rag.reranking.base import Reranker, RerankingError, RerankingResult


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        candidate_depth: int = 10,
        default_top_k: int = 5,
        model: Any | None = None,
        enabled: bool = True,
    ) -> None:
        _positive(candidate_depth, "candidate_depth")
        _positive(default_top_k, "top_k")
        if not isinstance(model_name, str) or not model_name.strip():
            raise RerankingError("model_name must not be empty")
        self.model_name = model_name
        self.candidate_depth = candidate_depth
        self.default_top_k = default_top_k
        self.enabled = bool(enabled)
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                raise RerankingError(f"Could not load cross-encoder: {exc}") from exc
        return self._model

    def rerank(
        self, query: str, candidates: Sequence[Any], top_k: int | None = None
    ) -> tuple[Any, ...]:
        if not isinstance(query, str) or not query.strip():
            raise RerankingError("query must be a non-empty string")
        if not isinstance(candidates, Sequence):
            raise RerankingError("candidates must be a sequence")
        selected_top_k = self.default_top_k if top_k is None else top_k
        _positive(selected_top_k, "top_k")
        values = tuple(candidates[: self.candidate_depth])
        if not values:
            return ()
        if not self.enabled:
            return values[:selected_top_k]
        pairs = []
        for candidate in values:
            content = getattr(candidate, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise RerankingError("each candidate must contain non-empty content")
            pairs.append((query.strip(), content))
        try:
            raw_scores = self._get_model().predict(pairs)
        except Exception as exc:
            raise RerankingError(f"Could not score candidates: {exc}") from exc
        try:
            scores = [float(score) for score in raw_scores]
        except (TypeError, ValueError) as exc:
            raise RerankingError("cross-encoder returned invalid scores") from exc
        if len(scores) != len(values) or not all(isfinite(score) for score in scores):
            raise RerankingError("cross-encoder returned invalid scores")
        ranked = sorted(
            zip(values, scores),
            key=lambda item: (-item[1], getattr(item[0], "rank", 0), item[0].chunk_id),
        )
        return tuple(
            RerankingResult(
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                content=candidate.content,
                metadata=dict(candidate.metadata),
                provenance=tuple(getattr(candidate, "provenance", ())),
                reranker_score=score,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(ranked[:selected_top_k], start=1)
        )


def _positive(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RerankingError(f"{name} must be greater than zero")
