import unittest

from enterprise_rag.evaluation.phase10 import build_comparison, identify_baseline


def row(qid, status="success", scoring="unscorable", sources=None, cited=None, latency=10):
    return {"dataset_name": "d", "question_id": qid, "execution": {"status": status, "latency_ms": latency}, "actual": {"sources": sources or [], "cited_sources": cited or []}, "trace": {"scoring_status": scoring}, "_path": qid}


class Phase10ComparisonTests(unittest.TestCase):
    def test_baseline_selects_one_newest_row_per_question(self):
        old = row("q1"); old["_path"] = "1_q1"
        new = row("q1"); new["_path"] = "2_q1"
        rows = [old, new] + [row(f"q{i}") for i in range(2, 61)]
        selected = identify_baseline(rows, 60)
        self.assertEqual(len(selected), 60)
        self.assertEqual(next(x for x in selected if x["question_id"] == "q1")["_path"], "2_q1")

    def test_newly_scorable_and_missing_values_are_not_zeroed(self):
        base = row("q1")
        current = row("q1", scoring="scored", sources=[{"chunk_id": "c"}], cited=[])
        report = build_comparison([base], [current], 1)
        item = report["per_question"][0]
        self.assertEqual(item["change_classification"], "newly_scorable")
        self.assertIsNone(item["baseline_retrieval_metrics"])
        self.assertEqual(report["phase10"]["successful"], 1)

    def test_failed_and_missing_question_classification(self):
        report = build_comparison([row("q1", status="failed")], [], 1)
        self.assertEqual(report["per_question"][0]["change_classification"], "unavailable")


if __name__ == "__main__":
    unittest.main()
