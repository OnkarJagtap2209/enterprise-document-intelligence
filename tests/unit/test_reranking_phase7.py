import unittest

from enterprise_rag.reranking import CrossEncoderReranker, RerankingError
from enterprise_rag.retrieval import HybridRetrievalResult


def candidate(cid, content, rank=1):
    return HybridRetrievalResult(cid, "doc", content, {"chunk_id": cid}, (), 0.1, {}, rank)


class FakeCrossEncoder:
    def __init__(self, scores): self.scores, self.calls = scores, []
    def predict(self, pairs): self.calls.append(pairs); return self.scores[:len(pairs)]


class RerankingTests(unittest.TestCase):
    def test_disabled_returns_phase6_candidates_unchanged(self):
        values = (candidate("a", "A"), candidate("b", "B"))
        result = CrossEncoderReranker(model=FakeCrossEncoder([2, 1]), enabled=False).rerank("q", values)
        self.assertEqual(result, values)

    def test_enabled_orders_scores_and_preserves_fields(self):
        values = (candidate("a", "A", 1), candidate("b", "B", 2))
        result = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.9]), default_top_k=1).rerank("q", values)
        self.assertEqual(result[0].chunk_id, "b")
        self.assertEqual(result[0].reranker_score, 0.9)
        self.assertEqual(result[0].rank, 1)
        self.assertEqual(result[0].metadata["chunk_id"], "b")

    def test_candidate_depth_empty_and_invalid_inputs(self):
        model = FakeCrossEncoder([0.5, 0.4])
        reranker = CrossEncoderReranker(model=model, candidate_depth=1)
        self.assertEqual(reranker.rerank("q", ()), ())
        self.assertEqual(len(reranker.rerank("q", (candidate("a", "A"), candidate("b", "B")))), 1)
        for query in ("", None):
            with self.assertRaises(RerankingError): reranker.rerank(query, ())

    def test_equal_scores_are_deterministic_by_original_rank_then_id(self):
        values = (candidate("b", "B", 1), candidate("a", "A", 2))
        result = CrossEncoderReranker(model=FakeCrossEncoder([1, 1])).rerank("q", values)
        self.assertEqual([item.chunk_id for item in result], ["b", "a"])


if __name__ == "__main__": unittest.main()
