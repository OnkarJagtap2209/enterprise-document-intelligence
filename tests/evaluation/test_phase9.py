import unittest
from enterprise_rag.evaluation import EvaluationQuestion, ExperimentRunner, ExperimentSpec, RelevanceJudgment
from enterprise_rag.retrieval import HybridRetrievalResult

class Retriever:
    def retrieve(self, query, top_k=None):
        return (HybridRetrievalResult("a", "d", "A", {"x": 1}, ({"p": 1},), .2, {}, 1), HybridRetrievalResult("b", "d", "B", {"x": 2}, ({"p": 2},), .1, {}, 2))[:top_k]

class Reranker:
    candidate_depth = 2
    def rerank(self, query, candidates, top_k=None): return tuple(reversed(candidates[:top_k]))

class Phase9Tests(unittest.TestCase):
    def test_baseline_and_reranked_runs_use_same_judgments(self):
        questions = [EvaluationQuestion("q", "query")]
        judgments = [RelevanceJudgment("q", "b", 1)]
        baseline = ExperimentRunner(Retriever(), ExperimentSpec("baseline", top_k=2)).run(questions, judgments)
        reranked = ExperimentRunner(Retriever(), ExperimentSpec("reranked", top_k=2), Reranker()).run(questions, judgments)
        self.assertEqual(baseline.traces[0].chunk_ids, ("a", "b"))
        self.assertEqual(reranked.traces[0].chunk_ids, ("b", "a"))
        self.assertGreater(reranked.summary["mrr"], baseline.summary["mrr"])

if __name__ == "__main__": unittest.main()
