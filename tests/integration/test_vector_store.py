import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import chromadb

from enterprise_rag.stores import (
    ChromaVectorStore,
    ChunkIndexer,
    VectorStoreError,
    chunk_metadata_to_chroma,
)


class FakeEmbeddingService:
    dimension = 3

    def embed_documents(self, texts):
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def metadata(index: int) -> dict:
    return {
        "chunk_id": f"chunk_{index:03d}",
        "document_id": "document-1",
        "source_filename": "report.pdf",
        "pages": [index + 1],
        "section_context": ["Results", f"Part {index}"],
        "content_type": "table" if index else "text",
        "chunk_index": index,
        "source_refs": [f"#/texts/{index}"],
        "structural_labels": ["text"],
    }


class VectorStoreTests(unittest.TestCase):
    def make_store(self):
        return ChromaVectorStore(
            db_path="unused",
            collection_name=f"phase4_{self._testMethodName}",
            embedding_model="fake-model",
            embedding_dimension=3,
            client=chromadb.EphemeralClient(),
        )

    def test_collection_creation_metadata_and_conversion(self) -> None:
        store = self.make_store()
        converted = chunk_metadata_to_chroma(metadata(0))
        stats = store.stats()

        self.assertEqual(stats.record_count, 0)
        self.assertEqual(stats.metadata["embedding_model"], "fake-model")
        self.assertEqual(stats.metadata["embedding_dimension"], 3)
        self.assertEqual(stats.configuration["hnsw"]["space"], "cosine")
        self.assertEqual(converted["pages_json"], "[1]")
        self.assertEqual(converted["section_path"], "Results > Part 0")
        self.assertEqual(json.loads(converted["source_refs_json"]), ["#/texts/0"])

    def test_upsert_is_idempotent_and_validates_dimensions(self) -> None:
        store = self.make_store()
        record_metadata = chunk_metadata_to_chroma(metadata(0))
        args = (["chunk_000"], [[1.0, 0.0, 0.0]], ["content"], [record_metadata])

        store.upsert(*args)
        store.upsert(*args)

        self.assertEqual(store.count(), 1)
        self.assertEqual(store.peek(1)["ids"], ["chunk_000"])
        with self.assertRaisesRegex(VectorStoreError, "dimension"):
            store.upsert(["bad"], [[1.0]], ["content"], [record_metadata])

    def test_chunk_artifact_indexes_in_batches_without_duplicates(self) -> None:
        artifact = {
            "artifact_schema": "enterprise_rag.chunks.v1",
            "chunks": [
                {"content": "Revenue rose.", "metadata": metadata(0)},
                {"content": "Margin table.", "metadata": metadata(1)},
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.chunks.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            store = self.make_store()
            indexer = ChunkIndexer(FakeEmbeddingService(), store, batch_size=1)

            first = indexer.index(path)
            second = indexer.index(path)

        self.assertEqual(first.chunk_count, 2)
        self.assertEqual(first.embeddings_created, 2)
        self.assertEqual(second.collection_record_count, 2)
        self.assertEqual(set(store.peek(5)["ids"]), {"chunk_000", "chunk_001"})
