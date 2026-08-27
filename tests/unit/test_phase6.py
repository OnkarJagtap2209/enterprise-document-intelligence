import unittest
import json

from enterprise_rag.retrieval import BM25Retriever, HybridRetriever, reciprocal_rank_fusion
from enterprise_rag.retrieval import SemanticRetriever
from enterprise_rag.stores import StoredVectorMatch


def chunk(identifier, text, index=0):
    return {
        "content": text,
        "metadata": {
            "chunk_id": identifier, "document_id": "doc", "source_filename": "x.pdf",
            "pages": [1], "section_context": ["Results"], "content_type": "text",
            "chunk_index": index, "source_refs": ["#/texts/0"], "structural_labels": ["text"],
        },
        "provenance": [{"source": "docling", "index": index}],
    }


class Phase6Tests(unittest.TestCase):
    def test_bm25_empty_corpus_rejected_and_zero_token_corpus_safe(self):
        with self.assertRaises(ValueError):
            BM25Retriever([])
        retriever = BM25Retriever([chunk("a", "!!!"), chunk("b", "---")])
        self.assertEqual(retriever.retrieve("revenue"), ())
        self.assertEqual(retriever.retrieve("!!!"), ())

    def test_bm25_ranks_matching_terms_and_preserves_provenance(self):
        retriever = BM25Retriever([chunk("a", "Revenue increased", 0), chunk("b", "Headcount increased", 1)])
        results = retriever.retrieve("revenue")
        self.assertEqual(results[0].chunk_id, "a")
        self.assertEqual(results[0].provenance[0]["source"], "docling")

    def test_rrf_uses_rank_contributions_and_fuses_duplicates(self):
        results = reciprocal_rank_fusion({"semantic": [("a", 1), ("b", 2)], "bm25": [("a", 1)]}, rrf_k=1)
        self.assertEqual(results[0].chunk_id, "a")
        self.assertAlmostEqual(results[0].rrf_score, 1.0)
        self.assertEqual(results[0].retriever_ranks, {"semantic": 1, "bm25": 1})

    def test_hybrid_returns_final_top_k_and_metadata(self):
        class Retriever:
            def __init__(self, values): self.values = values
            def retrieve(self, query, top_k=None): return tuple(self.values[:top_k])
        lexical = BM25Retriever([chunk("a", "Revenue", 0), chunk("b", "Employees", 1)])
        semantic = Retriever([type("R", (), {"chunk_id": "a", "document_id": "doc", "content": "Revenue", "metadata": {"chunk_id": "a"}, "rank": 1, "provenance": ()})()])
        result = HybridRetriever(semantic, lexical, default_top_k=1, candidate_depth=2).retrieve("revenue")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_id, "a")
        self.assertEqual(result[0].metadata["chunk_id"], "a")

    def test_rrf_equal_scores_are_sorted_by_chunk_id(self):
        results = reciprocal_rank_fusion({"r": [("b", 1), ("a", 1)]}, rrf_k=60)
        self.assertEqual([item.chunk_id for item in results], ["a", "b"])

    def test_semantic_provenance_round_trips_from_chroma_metadata(self):
        class Embedding:
            def embed_query(self, query): return [1.0]
        class Store:
            def semantic_query(self, embedding, top_k):
                return (StoredVectorMatch("a", "Revenue", {
                    "chunk_id": "a", "document_id": "doc",
                    "provenance_json": json.dumps([{"source": "docling"}]),
                }, 0.1),)
        result = SemanticRetriever(Embedding(), Store()).retrieve("revenue")[0]
        self.assertEqual(result.provenance, ({"source": "docling"},))


if __name__ == "__main__":
    unittest.main()
