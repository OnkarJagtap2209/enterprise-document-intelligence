"""Reusable local sentence embedding service."""

from __future__ import annotations

from math import isfinite
from typing import Any, Sequence


class EmbeddingError(ValueError):
    """Raised when text cannot be converted into valid embeddings."""


class SentenceTransformerEmbeddingService:
    """Embed documents and queries with one model loaded per instance."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise EmbeddingError("model_name must not be empty")
        if batch_size <= 0:
            raise EmbeddingError("batch_size must be greater than zero")
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name, local_files_only=True)
            except Exception as exc:
                raise EmbeddingError(f"Could not load embedding model: {exc}") from exc

        dimension_method = getattr(model, "get_embedding_dimension", None)
        if dimension_method is None:
            dimension_method = model.get_sentence_embedding_dimension
        dimension = dimension_method()
        if not isinstance(dimension, int) or dimension <= 0:
            raise EmbeddingError("Embedding model returned an invalid dimension")
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self._model = model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a non-empty batch of chunk text."""
        values = _validate_texts(texts)
        encode = getattr(self._model, "encode_document", None) or self._model.encode
        try:
            embeddings = encode(
                values,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Could not embed documents: {exc}") from exc
        return _validate_embeddings(embeddings, len(values), self.dimension)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query using the model's query-specific method when available."""
        value = _validate_texts([text])[0]
        encode = getattr(self._model, "encode_query", None) or self._model.encode
        try:
            embeddings = encode(
                [value],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise EmbeddingError(f"Could not embed query: {exc}") from exc
        return _validate_embeddings(embeddings, 1, self.dimension)[0]


def _validate_texts(texts: Sequence[str]) -> list[str]:
    if isinstance(texts, (str, bytes)) or not texts:
        raise EmbeddingError("At least one text is required")
    values = list(texts)
    if any(not isinstance(text, str) or not text.strip() for text in values):
        raise EmbeddingError("Embedding inputs must be non-empty strings")
    return values


def _validate_embeddings(
    embeddings: Any, expected_count: int, expected_dimension: int
) -> list[list[float]]:
    values = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    if not isinstance(values, list) or len(values) != expected_count:
        raise EmbeddingError("Embedding model returned an unexpected row count")
    validated: list[list[float]] = []
    for vector in values:
        if not isinstance(vector, (list, tuple)) or len(vector) != expected_dimension:
            raise EmbeddingError("Embedding model returned an unexpected dimension")
        try:
            row = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Embedding vector contains a non-numeric value") from exc
        if not all(isfinite(value) for value in row):
            raise EmbeddingError("Embedding vector contains a non-finite value")
        validated.append(row)
    return validated
