import unittest

from enterprise_rag.evaluation import (
    EvaluationQuestion, ExperimentRunner, ExperimentSpec, RelevanceJudgment,
    compute_metrics,
)


class Result:
    def __init__(self, chunk_id, rank): self.chunk_id, self.rank = chunk_id, rank


class Retriever:
    def retrieve(self, query, top_k=None):
        return (Result("a", 1), Result("b", 2))[:top_k]


class EvaluationTests(unittest.TestCase):
    def test_metrics_are_deterministic(self):
        self.assertEqual(compute_metrics(["a", "b"], {"a": 2}, 2)["mrr"], 1.0)

    def test_runner_produces_trace_and_summary(self):
        run = ExperimentRunner(Retriever(), ExperimentSpec("semantic")).run(
            [EvaluationQuestion("q1", "revenue")], [RelevanceJudgment("q1", "a", 1)]
        )
        self.assertEqual(run.traces[0].chunk_ids, ("a", "b"))
        self.assertEqual(run.summary["recall_at_k"], 1.0)


if __name__ == "__main__": unittest.main()
