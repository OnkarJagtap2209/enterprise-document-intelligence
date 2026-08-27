import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from enterprise_rag.ingestion import (
    DoclingPdfProcessor,
    ExtractionValidationError,
    IngestionPipeline,
    SourceDocumentError,
)


class FakeProcessor:
    def process(self, pdf_path: Path) -> dict:
        return {
            "schema_name": "DoclingDocument",
            "name": pdf_path.stem,
            "body": {"children": []},
            "pages": {"1": {"page_no": 1}},
            "texts": [
                {
                    "text": "Representative financial content",
                    "prov": [{"page_no": 1}],
                }
            ],
            "tables": [
                {
                    "data": {"num_rows": 1, "num_cols": 1, "table_cells": []},
                    "prov": [{"page_no": 1}],
                }
            ],
        }


class EmptyProcessor:
    def process(self, pdf_path: Path) -> dict:
        return {}


class FakeDocument:
    def export_to_dict(self) -> dict:
        return FakeProcessor().process(Path("sample.pdf"))


class FakeConverter:
    def convert(self, pdf_path: Path):
        return type("ConversionResult", (), {"document": FakeDocument()})()


class IngestionTests(unittest.TestCase):
    def test_nonexistent_pdf_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            pipeline = IngestionPipeline(directory, FakeProcessor())
            with self.assertRaisesRegex(SourceDocumentError, "does not exist"):
                pipeline.ingest(Path(directory) / "missing.pdf")

    def test_non_pdf_file_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "document.txt"
            source.write_text("not a PDF", encoding="utf-8")
            pipeline = IngestionPipeline(directory, FakeProcessor())
            with self.assertRaisesRegex(SourceDocumentError, "expected .pdf"):
                pipeline.ingest(source)

    def test_structured_artifact_is_persisted_with_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "financial-report.pdf"
            source.write_bytes(b"%PDF-1.4 test")
            output_dir = Path(directory) / "extracted"
            pipeline = IngestionPipeline(output_dir, FakeProcessor())

            with patch(
                "enterprise_rag.ingestion.pipeline.package_version",
                return_value="test-version",
            ):
                result = pipeline.ingest(source)

            artifact = json.loads(result.output_path.read_text(encoding="utf-8"))
            self.assertTrue(result.output_path.is_file())
            self.assertEqual(artifact["artifact_schema"], "enterprise_rag.extraction.v1")
            self.assertEqual(artifact["source"]["filename"], source.name)
            self.assertEqual(artifact["extraction"]["page_count"], 1)
            self.assertEqual(artifact["extraction"]["table_count"], 1)
            self.assertEqual(artifact["document"]["texts"][0]["prov"][0]["page_no"], 1)

    def test_empty_extraction_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "empty.pdf"
            source.write_bytes(b"%PDF-1.4 test")
            pipeline = IngestionPipeline(directory, EmptyProcessor())

            with self.assertRaisesRegex(ExtractionValidationError, "empty document"):
                pipeline.ingest(source)

    def test_docling_processor_preserves_exported_structure(self) -> None:
        document = DoclingPdfProcessor(FakeConverter()).process(Path("sample.pdf"))
        self.assertEqual(document["schema_name"], "DoclingDocument")
        self.assertEqual(document["tables"][0]["prov"][0]["page_no"], 1)

    @unittest.skipUnless(
        os.getenv("RUN_DOCLING_INTEGRATION") == "1",
        "set RUN_DOCLING_INTEGRATION=1 to process the representative PDF",
    )
    def test_representative_infosys_pdf(self) -> None:
        source = Path("data/documents/q1-26-2027.pdf")
        with TemporaryDirectory() as directory:
            result = IngestionPipeline(directory).ingest(source)
            artifact = json.loads(result.output_path.read_text(encoding="utf-8"))

        self.assertGreater(result.page_count, 0)
        self.assertGreater(result.text_item_count, 0)
        self.assertEqual(artifact["source"]["filename"], source.name)
        self.assertTrue(artifact["document"]["pages"])
