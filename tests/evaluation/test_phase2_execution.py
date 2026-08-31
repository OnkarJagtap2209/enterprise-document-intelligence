import json
import tempfile
import unittest
from pathlib import Path
from enterprise_rag.evaluation import discover_datasets, execute_one, persist_execution

class Outcome:
    answer = "answer"; request_id = "req"
    sources = ()
class Service:
    def query(self, query): return Outcome()
class Failing:
    def query(self, query): raise RuntimeError("temporary failure")

class Phase2ExecutionTests(unittest.TestCase):
    def test_success_is_captured_and_persisted(self):
        dataset = discover_datasets("data/evaluation/questions")[0]
        result = execute_one(dataset, dataset.questions[0], Service())
        self.assertEqual(result.execution["status"], "success")
        with tempfile.TemporaryDirectory() as d:
            path = persist_execution(result, d)
            self.assertTrue(Path(path).is_file())
            self.assertEqual(json.loads(Path(path).read_text())["question_id"], dataset.questions[0].question_id)

    def test_failure_is_recorded_without_answer(self):
        dataset = discover_datasets("data/evaluation/questions")[0]
        result = execute_one(dataset, dataset.questions[0], Failing())
        self.assertEqual(result.execution["status"], "failed")
        self.assertIsNone(result.actual["answer"])
        self.assertEqual(result.execution["error"]["type"], "RuntimeError")

if __name__ == "__main__": unittest.main()
