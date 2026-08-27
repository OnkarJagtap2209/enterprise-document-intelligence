"""Orchestration and persistence for structured PDF ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version as package_version
import json
import logging
from pathlib import Path
from typing import Any

from enterprise_rag.ingestion.pdf_processor import DoclingPdfProcessor
from enterprise_rag.ingestion.validator import (
    validate_pdf_path,
    validate_structured_document,
)

logger = logging.getLogger(__name__)


class ArtifactPersistenceError(RuntimeError):
    """Raised when an extraction artifact cannot be written."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Small summary returned after a successful ingestion."""

    document_id: str
    source_path: Path
    output_path: Path
    page_count: int
    text_item_count: int
    table_count: int


class IngestionPipeline:
    """Validate, convert, validate, and persist one PDF."""

    def __init__(
        self,
        output_dir: str | Path,
        processor: DoclingPdfProcessor | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self._processor = processor

    def ingest(self, pdf_path: str | Path) -> IngestionResult:
        source_path = validate_pdf_path(pdf_path)
        logger.info("Starting PDF ingestion: %s", source_path.name)

        processor = self._processor or DoclingPdfProcessor()
        document = processor.process(source_path)
        validate_structured_document(document)

        document_id = _sha256_file(source_path)
        page_count = _collection_size(document.get("pages"))
        text_item_count = _collection_size(document.get("texts"))
        table_count = _collection_size(document.get("tables"))
        output_path = self.output_dir / (
            f"{source_path.stem}.{document_id[:12]}.docling.json"
        )
        artifact = {
            "artifact_schema": "enterprise_rag.extraction.v1",
            "document_id": document_id,
            "source": {
                "filename": source_path.name,
                "path": str(source_path),
                "sha256": document_id,
            },
            "extraction": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "extractor": "docling",
                "extractor_version": package_version("docling"),
                "page_count": page_count,
                "text_item_count": text_item_count,
                "table_count": table_count,
            },
            "document": document,
        }
        _persist_artifact(artifact, output_path)

        logger.info(
            "PDF ingestion completed: pages=%d text_items=%d tables=%d output=%s",
            page_count,
            text_item_count,
            table_count,
            output_path,
        )
        return IngestionResult(
            document_id=document_id,
            source_path=source_path,
            output_path=output_path.resolve(),
            page_count=page_count,
            text_item_count=text_item_count,
            table_count=table_count,
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection_size(value: Any) -> int:
    return len(value) if isinstance(value, (dict, list)) else 0


def _persist_artifact(artifact: dict[str, Any], output_path: Path) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(artifact, output, ensure_ascii=False)
        temporary_path.replace(output_path)
    except (OSError, TypeError, ValueError) as exc:
        temporary_path.unlink(missing_ok=True)
        raise ArtifactPersistenceError(
            f"Could not persist extraction artifact to {output_path}: {exc}"
        ) from exc
