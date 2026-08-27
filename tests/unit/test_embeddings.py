import unittest

from enterprise_rag.embeddings import (
    EmbeddingError,
    SentenceTransformerEmbeddingService,
)


class FakeModel:
    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode_document(self, texts, **kwargs):
        self.document_calls += 1
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def encode_query(self, texts, **kwargs):
        self.query_calls += 1
        return [[float(len(texts[0])), 0.0, 1.0]]


class EmbeddingServiceTests(unittest.TestCase):
    def test_document_and_query_embeddings_have_predictable_dimensions(self) -> None:
        model = FakeModel()
        service = SentenceTransformerEmbeddingService("fake-model", 2, model=model)

        documents = service.embed_documents(["one", "two words"])
        query = service.embed_query("question")

        self.assertEqual(service.dimension, 3)
        self.assertEqual(len(documents), 2)
        self.assertTrue(all(len(vector) == 3 for vector in documents))
        self.assertEqual(len(query), 3)
        self.assertEqual(model.document_calls, 1)
        self.assertEqual(model.query_calls, 1)

    def test_empty_and_blank_input_are_rejected(self) -> None:
        service = SentenceTransformerEmbeddingService("fake-model", model=FakeModel())
        with self.assertRaisesRegex(EmbeddingError, "At least one"):
            service.embed_documents([])
        with self.assertRaisesRegex(EmbeddingError, "non-empty strings"):
            service.embed_documents([" "])
        with self.assertRaisesRegex(EmbeddingError, "non-empty strings"):
            service.embed_query("")

    def test_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(EmbeddingError, "batch_size"):
            SentenceTransformerEmbeddingService("fake", 0, FakeModel())
