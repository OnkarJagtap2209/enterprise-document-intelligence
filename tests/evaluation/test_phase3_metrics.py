import unittest
from enterprise_rag.evaluation import compute_retrieval_metrics, aggregate_metrics

class Phase3MetricTests(unittest.TestCase):
    def test_standard_metrics_and_first_rank(self):
        values = compute_retrieval_metrics(["x", "y", "r", "z", "q"], {"r"})
        self.assertEqual(values["first_relevant_rank"], 3)
        self.assertEqual(values["hit_at_1"], 0.0)
        self.assertEqual(values["hit_at_3"], 1.0)
        self.assertAlmostEqual(values["precision_at_3"], 1/3)
        self.assertEqual(values["recall_at_5"], 1.0)
        self.assertAlmostEqual(values["reciprocal_rank"], 1/3)

    def test_empty_ground_truth_and_aggregation(self):
        self.assertEqual(compute_retrieval_metrics(["x"], set())["reciprocal_rank"], 0.0)
        summary = aggregate_metrics([{"hit_at_1": 1.0}, {"hit_at_1": 0.0}])
        self.assertEqual(summary["successful_questions"], 2)
        self.assertEqual(summary["hit_at_1"], 0.5)

if __name__ == "__main__": unittest.main()
