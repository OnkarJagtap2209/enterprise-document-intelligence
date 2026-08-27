"""Serializable document chunk models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    chunk_id: str
    document_id: str
    source_filename: str
    pages: tuple[int, ...]
    section_context: tuple[str, ...]
    content_type: str
    chunk_index: int
    source_refs: tuple[str, ...]
    structural_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    content: str
    metadata: ChunkMetadata
    provenance: tuple[dict[str, Any], ...]
    structured_content: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "metadata": asdict(self.metadata),
            "provenance": list(self.provenance),
            "structured_content": self.structured_content,
        }
