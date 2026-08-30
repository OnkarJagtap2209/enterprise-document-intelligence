"""Deterministic Okapi BM25 retrieval over persisted Phase 3 chunks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from enterprise_rag.stores.indexer import load_chunk_artifact
from enterprise_rag.stores.vector_store import chunk_metadata_to_chroma
from enterprise_rag.routing import QueryConstraints, matches_constraints

_TOKEN_PATTERN = re.compile(
    r"(?:[$€£₹])?\d+(?:[.,]\d+)*%?|[^\W_]+(?:[./'-][^\W_]+)*",
    re.UNICODE,
)


class BM25RetrievalError(ValueError):
    """Raised when a BM25 corpus or query is invalid."""


@dataclass(frozen=True, slots=True)
class BM25RetrievalResult:
    """One lexical match; score is an unnormalized Okapi BM25 score."""

    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    bm25_score: float
    rank: int
    provenance: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class _BM25Document:
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any]
    term_frequencies: Counter[str]
    length: int
    corpus_position: int
    provenance: tuple[dict[str, Any], ...]


class BM25Retriever:
    """Build a small in-memory BM25 corpus with stable Phase 3 chunk identities."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        default_top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        _validate_top_k(default_top_k)
        if k1 <= 0:
            raise BM25RetrievalError("k1 must be greater than zero")
        if b < 0 or b > 1:
            raise BM25RetrievalError("b must be between zero and one")
        if not chunks:
            raise BM25RetrievalError("BM25 corpus must contain at least one chunk")

        documents: list[_BM25Document] = []
        document_frequencies: Counter[str] = Counter()
        seen_ids: set[str] = set()
        for position, chunk in enumerate(chunks):
            document = _build_document(chunk, position)
            if document.chunk_id in seen_ids:
                raise BM25RetrievalError(
                    f"Duplicate chunk_id in BM25 corpus: {document.chunk_id}"
                )
            seen_ids.add(document.chunk_id)
            documents.append(document)
            document_frequencies.update(document.term_frequencies.keys())

        self._documents = tuple(documents)
        self._document_frequencies = document_frequencies
        self._average_length = sum(doc.length for doc in documents) / len(documents)
        self.default_top_k = default_top_k
        self.k1 = float(k1)
        self.b = float(b)

    @classmethod
    def from_chunk_artifact(
        cls,
        artifact_path: str | Path,
        default_top_k: int = 10,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Retriever:
        """Load the canonical Phase 3 artifact directly; no second corpus is stored."""
        _, chunks = load_chunk_artifact(artifact_path)
        return cls(chunks, default_top_k=default_top_k, k1=k1, b=b)

    @property
    def corpus_size(self) -> int:
        return len(self._documents)

    def retrieve(
        self, query: str, top_k: int | None = None, metadata_filter: QueryConstraints | None = None
    ) -> tuple[BM25RetrievalResult, ...]:
        if not isinstance(query, str) or not query.strip():
            raise BM25RetrievalError("query must be a non-empty string")
        selected_top_k = self.default_top_k if top_k is None else top_k
        _validate_top_k(selected_top_k)
        query_terms = tokenize_financial_text(query)
        if not query_terms:
            return ()

        scored = [
            (self._score(document, query_terms), document)
            for document in self._documents
            if metadata_filter is None or matches_constraints(document.metadata, metadata_filter)
        ]
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: (-item[0], item[1].corpus_position))
        return tuple(
            BM25RetrievalResult(
                chunk_id=document.chunk_id,
                document_id=document.document_id,
                content=document.content,
                metadata=document.metadata,
                bm25_score=float(score),
                rank=rank,
                provenance=document.provenance,
            )
            for rank, (score, document) in enumerate(
                scored[:selected_top_k], start=1
            )
        )

    def _score(self, document: _BM25Document, query_terms: Sequence[str]) -> float:
        if self._average_length == 0:
            return 0.0
        score = 0.0
        corpus_size = len(self._documents)
        length_ratio = document.length / self._average_length
        for term in query_terms:
            frequency = document.term_frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = log(
                1 + (corpus_size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length_ratio
            )
            score += inverse_document_frequency * (
                frequency * (self.k1 + 1) / denominator
            )
        return score


def tokenize_financial_text(text: str) -> tuple[str, ...]:
    """Case-fold text while retaining financial numbers and punctuation."""
    if not isinstance(text, str):
        raise BM25RetrievalError("text to tokenize must be a string")
    return tuple(_normalize_token(token.casefold()) for token in _TOKEN_PATTERN.findall(text))


def _normalize_token(token: str) -> str:
    """Apply a minimal, deterministic singular/plural normalization."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _build_document(chunk: Mapping[str, Any], position: int) -> _BM25Document:
    content = chunk.get("content")
    metadata = chunk.get("metadata")
    if not isinstance(content, str) or not content.strip():
        raise BM25RetrievalError(f"Chunk {position} has no content")
    if not isinstance(metadata, Mapping):
        raise BM25RetrievalError(f"Chunk {position} has no metadata")
    try:
        converted_metadata = chunk_metadata_to_chroma(metadata, provenance=chunk.get("provenance"))
    except (TypeError, ValueError) as exc:
        raise BM25RetrievalError(f"Chunk {position} has invalid metadata: {exc}") from exc
    tokens = tokenize_financial_text(content)
    return _BM25Document(
        chunk_id=converted_metadata["chunk_id"],
        document_id=converted_metadata["document_id"],
        content=content,
        metadata=converted_metadata,
        term_frequencies=Counter(tokens),
        length=len(tokens),
        corpus_position=position,
        provenance=_provenance_values(chunk.get("provenance")),
    )


def _provenance_values(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _validate_top_k(top_k: int) -> None:
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise BM25RetrievalError("top_k must be greater than zero")
