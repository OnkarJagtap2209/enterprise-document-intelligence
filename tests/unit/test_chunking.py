import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from enterprise_rag.ingestion import (
    ChunkingError,
    ChunkingPipeline,
    StructureAwareChunker,
    load_extraction_artifact,
)


def extraction_artifact() -> dict:
    provenance_page_1 = [{"page_no": 1, "bbox": {"l": 1, "t": 2}}]
    provenance_page_2 = [{"page_no": 2, "bbox": {"l": 3, "t": 4}}]
    return {
        "artifact_schema": "enterprise_rag.extraction.v1",
        "document_id": "a" * 64,
        "source": {"filename": "report.pdf", "path": "report.pdf"},
        "document": {
            "body": {
                "children": [
                    {"$ref": "#/texts/0"},
                    {"$ref": "#/texts/1"},
                    {"$ref": "#/groups/0"},
                    {"$ref": "#/tables/0"},
                ]
            },
            "texts": [
                {
                    "label": "section_header",
                    "text": "Financial Results",
                    "prov": provenance_page_1,
                },
                {
                    "label": "text",
                    "text": "Revenue increased during the quarter.",
                    "prov": provenance_page_1,
                },
                {
                    "label": "list_item",
                    "text": "Long financial explanation " * 20,
                    "prov": provenance_page_1,
                },
            ],
            "groups": [
                {"children": [{"$ref": "#/texts/2"}]},
            ],
            "pictures": [],
            "tables": [
                {
                    "label": "table",
                    "prov": provenance_page_2,
                    "data": {
                        "num_rows": 2,
                        "num_cols": 2,
                        "grid": [
                            [
                                {"text": "Metric", "column_header": True},
                                {"text": "Value", "column_header": True},
                            ],
                            [
                                {"text": "Revenue", "row_header": True},
                                {"text": "100"},
                            ],
                        ],
                    },
                }
            ],
        },
    }


class StructureAwareChunkingTests(unittest.TestCase):
    def test_chunks_are_deterministic_and_preserve_structure(self) -> None:
        chunker = StructureAwareChunker(max_chars=140, overlap_chars=20)

        first = chunker.chunk(extraction_artifact())
        second = chunker.chunk(extraction_artifact())

        self.assertEqual(
            [chunk.metadata.chunk_id for chunk in first],
            [chunk.metadata.chunk_id for chunk in second],
        )
        self.assertEqual(
            [chunk.metadata.chunk_index for chunk in first], list(range(len(first)))
        )
        self.assertTrue(all(chunk.content.strip() for chunk in first))
        self.assertTrue(all(len(chunk.content) <= 140 for chunk in first))
        self.assertTrue(all(chunk.metadata.document_id == "a" * 64 for chunk in first))
        self.assertTrue(all(chunk.metadata.source_filename == "report.pdf" for chunk in first))

        oversized = [
            chunk for chunk in first if "#/texts/2" in chunk.metadata.source_refs
        ]
        self.assertGreater(len(oversized), 1)
        self.assertTrue(all(chunk.metadata.pages == (1,) for chunk in oversized))
        self.assertTrue(
            all(chunk.metadata.section_context == ("Financial Results",) for chunk in oversized)
        )

        tables = [chunk for chunk in first if chunk.metadata.content_type == "table"]
        self.assertTrue(tables)
        self.assertTrue(all(chunk.metadata.pages == (2,) for chunk in tables))
        self.assertEqual(tables[0].structured_content["num_rows"], 2)
        self.assertTrue(tables[0].structured_content["rows"])
        self.assertEqual(tables[0].provenance[0]["page_no"], 2)

    def test_artifact_can_be_loaded_chunked_and_persisted(self) -> None:
        with TemporaryDirectory() as directory:
            extraction_path = Path(directory) / "report.docling.json"
            extraction_path.write_text(
                json.dumps(extraction_artifact()), encoding="utf-8"
            )
            loaded_path, loaded = load_extraction_artifact(extraction_path)
            pipeline = ChunkingPipeline(
                Path(directory) / "chunks",
                StructureAwareChunker(max_chars=200, overlap_chars=20),
            )

            result = pipeline.run(loaded_path)
            persisted = json.loads(result.output_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["document_id"], "a" * 64)
        self.assertEqual(persisted["artifact_schema"], "enterprise_rag.chunks.v1")
        self.assertEqual(persisted["chunking"]["chunk_count"], len(result.chunks))
        self.assertTrue(persisted["chunks"][0]["metadata"]["chunk_id"])

    def test_invalid_or_empty_input_is_rejected(self) -> None:
        artifact = extraction_artifact()
        artifact["document"]["body"]["children"] = []
        with self.assertRaisesRegex(ChunkingError, "no chunkable content"):
            StructureAwareChunker().chunk(artifact)

        with TemporaryDirectory() as directory:
            malformed = Path(directory) / "bad.json"
            malformed.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(ChunkingError, "Could not load"):
                load_extraction_artifact(malformed)

    def test_invalid_chunk_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ChunkingError, "overlap_chars"):
            StructureAwareChunker(max_chars=100, overlap_chars=100)
