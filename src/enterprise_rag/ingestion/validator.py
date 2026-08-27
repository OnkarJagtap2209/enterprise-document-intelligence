"""Validation for PDF sources and Docling extraction results."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SourceDocumentError(ValueError):
    """Raised when an ingestion source is not a readable PDF file."""


class ExtractionValidationError(ValueError):
    """Raised when Docling returns no usable structured content."""


def validate_pdf_path(pdf_path: str | Path) -> Path:
    """Return a resolved PDF path or raise a useful validation error."""
    path = Path(pdf_path).expanduser()
    if not path.exists():
        raise SourceDocumentError(f"PDF does not exist: {path}")
    if not path.is_file():
        raise SourceDocumentError(f"PDF path is not a file: {path}")
    if path.suffix.lower() != ".pdf":
        raise SourceDocumentError(f"Unsupported file type (expected .pdf): {path}")
    return path.resolve()


def validate_structured_document(document: Mapping[str, Any]) -> None:
    """Reject malformed or content-free Docling document exports."""
    if not document:
        raise ExtractionValidationError("Docling returned an empty document")

    content_keys = ("texts", "tables", "pictures", "key_value_items", "form_items")
    if not any(document.get(key) for key in content_keys):
        raise ExtractionValidationError(
            "Docling returned no text, tables, pictures, key-value items, or forms"
        )
