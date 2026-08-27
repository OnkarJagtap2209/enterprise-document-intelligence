import unittest

import chromadb

from enterprise_rag.retrieval import SemanticRetriever
from enterprise_rag.stores import ChromaVectorStore, chunk_metadata_to_chroma


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.query_count = 0

    def embed_query(self, query):
        self.query_count += 1
        return [1.0, 0.0, 0.0]


def metadata(index: int) -> dict:
    return {
        "chunk_id": f"semantic_chunk_{index}",
        "document_id": "semantic-document",
        "source_filename": "semantic-report.pdf",
        "pages": [index + 1],
        "section_context": ["Quarterly results"],
        "content_type": "text",
        "chunk_index": index,
        "source_refs": [f"#/texts/{index}"],
        "structural_labels": ["text"],
    }


class SemanticRetrievalIntegrationTests(unittest.TestCase):
    def make_store(self, suffix: str) -> ChromaVectorStore:
        return ChromaVectorStore(
            db_path="unused",
            collection_name=f"phase5_{suffix}",
            embedding_model="fake-model",
            embedding_dimension=3,
            client=chromadb.EphemeralClient(),
        )

    def test_chroma_query_returns_ranked_content_metadata_and_distance(self) -> None:
        store = self.make_store("ranked")
        store.upsert(
            ids=["semantic_chunk_0", "semantic_chunk_1"],
            embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            documents=["Revenue increased.", "Employee count increased."],
            metadatas=[
                chunk_metadata_to_chroma(metadata(0)),
                chunk_metadata_to_chroma(metadata(1)),
            ],
        )
        embedding_service = FakeEmbeddingService()

        results = SemanticRetriever(embedding_service, store).retrieve(
            "revenue", top_k=5
        )

        self.assertEqual(embedding_service.query_count, 1)
        self.assertEqual(len(results), 2)
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(results[0].chunk_id, "semantic_chunk_0")
        self.assertEqual(results[0].content, "Revenue increased.")
        self.assertEqual(results[0].metadata["pages_json"], "[1]")
        self.assertAlmostEqual(results[0].distance, 0.0, places=6)
        self.assertLessEqual(results[0].distance, results[1].distance)

    def test_empty_chroma_collection_returns_no_results(self) -> None:
        store = self.make_store("empty")
        results = SemanticRetriever(FakeEmbeddingService(), store).retrieve("revenue")
        self.assertEqual(results, ())
