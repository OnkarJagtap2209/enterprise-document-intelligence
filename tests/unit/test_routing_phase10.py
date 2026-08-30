import unittest
from enterprise_rag.routing import QueryConstraints, matches_constraints, understand_query

class RoutingTests(unittest.TestCase):
    def test_explicit_year_and_type(self):
        result = understand_query("Show table revenue for 2027")
        self.assertEqual(result.constraints, QueryConstraints(content_type="table", year=2027))
        self.assertFalse(result.clarification_required)

    def test_financial_period_years_do_not_request_clarification(self):
        result = understand_query("Compare revenue in 2026 and 2027")
        self.assertFalse(result.clarification_required)
        self.assertIsNone(result.constraints.year)

    def test_conflicting_document_years_request_clarification(self):
        result = understand_query("Compare the 2026 annual report with the 2027 annual report")
        self.assertTrue(result.clarification_required)
        self.assertIsNone(result.retrieval_query)

    def test_metadata_matching_is_explicit_and_deterministic(self):
        constraints = QueryConstraints(source_filename="q1-26-2027.pdf", content_type="table")
        metadata = {"source_filename": "q1-26-2027.pdf", "content_type": "table"}
        self.assertTrue(matches_constraints(metadata, constraints))
        self.assertFalse(matches_constraints({"source_filename": "other.pdf", "content_type": "table"}, constraints))

    def test_period_year_does_not_use_filename_substring(self):
        self.assertTrue(matches_constraints(
            {"source_filename": "q1-26-2027.pdf"}, QueryConstraints(year=2026)
        ))
        self.assertFalse(matches_constraints(
            {"source_filename": "q1-26-2027.pdf", "period_year": 2027}, QueryConstraints(year=2026)
        ))

if __name__ == "__main__": unittest.main()
