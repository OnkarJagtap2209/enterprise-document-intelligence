import json
import tempfile
import unittest
from pathlib import Path

from enterprise_rag.evaluation import EvaluationDatasetError, discover_datasets, load_dataset


class DatasetPhase1Tests(unittest.TestCase):
    def test_all_datasets_and_rich_fields_load(self):
        datasets = discover_datasets("data/evaluation/questions")
        self.assertEqual(len(datasets), 5)
        self.assertEqual([len(d.questions) for d in datasets], [12] * 5)
        self.assertEqual(sum(len(d.questions) for d in datasets), 60)
        question = next(q for d in datasets for q in d.questions if q.relevant_pages)
        self.assertTrue(question.question_id)
        self.assertTrue(question.expected_answer)
        self.assertTrue(question.expected_facts)
        self.assertTrue(question.relevant_pages)
        self.assertTrue(datasets[0].source_document["filename"])

    def test_malformed_and_duplicate_question_ids_rejected(self):
        payload = {"dataset_version": "v1", "dataset_name": "x", "source_document": {"filename": "x.pdf"}, "questions": [{"question_id": "q1", "question": "a"}, {"question_id": "q1", "question": "b"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(EvaluationDatasetError): load_dataset(path)


if __name__ == "__main__": unittest.main()
