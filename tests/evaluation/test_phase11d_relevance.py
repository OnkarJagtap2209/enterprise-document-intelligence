import unittest

from enterprise_rag.evaluation import EvaluationQuestion
from enterprise_rag.evaluation.relevance import source_is_relevant


class Phase11DRelevanceTests(unittest.TestCase):
    def test_wrong_document_with_generic_keyword_overlap_is_irrelevant(self):
        question = EvaluationQuestion(
            question_id="q1",
            query="What were revenues in Q1 FY2027?",
            retrieval_targets=("Revenues", "quarter"),
            document="q1-26-2027(1).pdf",
        )
        source = {
            "source_filename": "form20f-25-2026.pdf",
            "content": "Revenues and financial results for the year increased.",
        }
        self.assertFalse(source_is_relevant(source, question))

    def test_known_source_alias_remains_relevant(self):
        question = EvaluationQuestion(
            question_id="q1",
            query="What were revenues in Q1 FY2027?",
            retrieval_targets=("Revenues",),
            document="q1-26-2027(1).pdf",
        )
        source = {
            "source_filename": "q1-26-2027.pdf",
            "content": "Revenues were reported for the quarter.",
        }
        self.assertTrue(source_is_relevant(source, question))

    def test_keyword_fallback_remains_for_missing_source_identity(self):
        question = EvaluationQuestion(
            question_id="q1",
            query="What were revenues?",
            retrieval_targets=("Revenues",),
        )
        source = {"content": "Revenues were reported."}
        self.assertTrue(source_is_relevant(source, question))


if __name__ == "__main__":
    unittest.main()
