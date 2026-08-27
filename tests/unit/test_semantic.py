import unittest

from enterprise_rag.retrieval import SemanticRetrievalError, SemanticRetriever
from enterprise_rag.stores import StoredVectorMatch


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.queries = []

    def embed_query(self, query):
        self.queries.append(query)
        return [1.0, 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, matches=()) -> None:
        self.matches = tuple(matches)
        self.calls = []

    def semantic_query(self, embedding, top_k):
        self.calls.append((embedding, top_k))
        return self.matches[:top_k]


def match(index: int, distance: float) -> StoredVectorMatch:
    return StoredVectorMatch(
        chunk_id=f"chunk_{index}",
        document=f"Content {index}",
        metadata={
            "chunk_id": f"chunk_{index}",
            "document_id": "document-1",
            "source_filename": "report.pdf",
            "pages_json": f"[{index + 1}]",
            "section_context_json": '["Results"]',
            "source_refs_json": f'["#/texts/{index}"]',
            "structural_labels_json": '["text"]',
            "content_type": "text",
            "chunk_index": index,
        },
        distance=distance,
    )


class SemanticRetrieverTests(unittest.TestCase):
    def test_query_uses_existing_services_and_preserves_order(self) -> None:
        embedding_service = FakeEmbeddingService()
        vector_store = FakeVectorStore([match(0, 0.1), match(1, 0.25)])
        retriever = SemanticRetriever(embedding_service, vector_store, default_top_k=2)

        results = retriever.retrieve("  quarterly revenue  ")

        self.assertEqual(embedding_service.queries, ["quarterly revenue"])
        self.assertEqual(vector_store.calls, [([1.0, 0.0, 0.0], 2)])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual([result.chunk_id for result in results], ["chunk_0", "chunk_1"])
        self.assertEqual([result.distance for result in results], [0.1, 0.25])
        self.assertEqual(results[0].content, "Content 0")
        self.assertEqual(results[0].metadata["pages_json"], "[1]")

    def test_top_k_and_available_result_count_are_respected(self) -> None:
        retriever = SemanticRetriever(
            FakeEmbeddingService(), FakeVectorStore([match(0, 0.1)]), default_top_k=5
        )

        results = retriever.retrieve("revenue", top_k=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rank, 1)

    def test_empty_collection_returns_no_results(self) -> None:
        retriever = SemanticRetriever(FakeEmbeddingService(), FakeVectorStore())
        self.assertEqual(retriever.retrieve("revenue"), ())

    def test_invalid_query_and_top_k_are_rejected(self) -> None:
        retriever = SemanticRetriever(FakeEmbeddingService(), FakeVectorStore())
        for query in ("", "   ", None):
            with self.subTest(query=query):
                with self.assertRaisesRegex(SemanticRetrievalError, "query"):
                    retriever.retrieve(query)
        for top_k in (0, -1, 1.5, True):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(SemanticRetrievalError, "top_k"):
                    retriever.retrieve("revenue", top_k=top_k)
