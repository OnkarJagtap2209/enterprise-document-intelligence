import unittest

from enterprise_rag.application.rag_service import RAGService
from enterprise_rag.generation.context_builder import ContextItem
from enterprise_rag.generation.gemini import GenerationResult


class _Retrieved:
    chunk_id = "chunk-1"
    document_id = "doc-1"
    content = "Revenue was 48,211."
    metadata = {"source_filename": "report.pdf", "page_start": 3}
    provenance = ()


class _Retriever:
    def retrieve(self, query, metadata_filter=None):
        return (_Retrieved(),)


class _Generator:
    def generate(self, question, results):
        item = ContextItem(
            "chunk-1", "doc-1", "Revenue was 48,211.",
            {"source_filename": "report.pdf", "page_start": 3}, (),
        )
        return GenerationResult("Revenue was 48,211.", ()), (item,)


class Phase9EvidenceTests(unittest.TestCase):
    def test_context_is_exposed_for_evaluation_while_api_sources_remain_cited_only(self):
        outcome = RAGService(_Retriever(), _Generator()).query("What was revenue?")
        self.assertEqual(outcome.sources, ())
        self.assertEqual(tuple(item.chunk_id for item in outcome.retrieved_sources), ("chunk-1",))


if __name__ == "__main__":
    unittest.main()
