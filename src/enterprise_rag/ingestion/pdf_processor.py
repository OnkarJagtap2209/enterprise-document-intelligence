"""Docling-specific PDF conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PdfProcessingError(RuntimeError):
    """Raised when Docling cannot produce a structured document."""


class DoclingPdfProcessor:
    """Convert a PDF into Docling's lossless dictionary representation."""

    def __init__(self, converter: Any | None = None) -> None:
        if converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter(allowed_formats=[InputFormat.PDF])
        self._converter = converter

    def process(self, pdf_path: Path) -> dict[str, Any]:
        try:
            result = self._converter.convert(pdf_path)
            document = result.document.export_to_dict()
        except Exception as exc:
            raise PdfProcessingError(
                f"Docling failed to extract {pdf_path.name}: {exc}"
            ) from exc

        if not isinstance(document, dict):
            raise PdfProcessingError(
                f"Docling returned an unexpected result for {pdf_path.name}"
            )
        return document
