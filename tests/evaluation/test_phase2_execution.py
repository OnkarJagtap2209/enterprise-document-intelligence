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

class Retrieved:
    chunk_id = "c"; document_id = "d"; content = "e"; metadata = {"source_filename": "x.pdf"}; provenance = ()
class RichOutcome(Outcome):
    retrieved_sources = (Retrieved(),)
class RichService:
    def query(self, query): return RichOutcome()
class FilterService:
    source_filename = None
    def query(self, query, source_filename=None):
        self.source_filename = source_filename
        return Outcome()

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

    def test_retrieved_sources_are_preserved_separately_from_citations(self):
        dataset = discover_datasets("data/evaluation/questions")[0]
        result = execute_one(dataset, dataset.questions[0], RichService())
        self.assertEqual(result.actual["sources"][0]["chunk_id"], "c")
        self.assertEqual(result.actual["cited_sources"], [])

    def test_dataset_source_filename_is_propagated_when_supported(self):
        dataset = discover_datasets("data/evaluation/questions")[0]
        service = FilterService()
        execute_one(dataset, dataset.questions[0], service)
        self.assertEqual(service.source_filename, dataset.source_document["filename"])

if __name__ == "__main__": unittest.main()
