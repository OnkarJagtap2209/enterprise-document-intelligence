import unittest
from enterprise_rag.evaluation import evaluate_answer

FACTS = ("revenue=48211", "growth=14.0%")
SOURCES = [{"content": "Revenues 48,211 crore; growth 14.0%", "source_filename": "x.pdf"}]

class Phase4AnswerTests(unittest.TestCase):
    def test_correct_and_grounded(self):
        result = evaluate_answer("Revenue was 48,211 and growth was 14.0%.", FACTS, SOURCES)
        self.assertEqual(result.correctness_status, "correct")
        self.assertEqual(result.groundedness_status, "grounded")

    def test_partial_and_no_evidence(self):
        result = evaluate_answer("Revenue was 48,211.", FACTS, SOURCES)
        self.assertEqual(result.correctness_status, "partially_correct")
        self.assertEqual(result.groundedness_status, "partially_grounded")
        self.assertEqual(evaluate_answer("x", FACTS, []).evaluator_status, "unscorable")

    def test_empty_and_failed_answer_are_unscorable(self):
        self.assertEqual(evaluate_answer(None, FACTS, []).correctness_status, "unscorable")
        self.assertEqual(evaluate_answer("x", (), SOURCES).unscorable_reason, "missing expected facts")

if __name__ == "__main__": unittest.main()
