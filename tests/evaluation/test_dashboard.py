import json

from enterprise_rag.ui.evaluation_dashboard import aggregate_results, load_current_results


def _row(dataset="Infosys Q1 FY2027 Evaluation Set", question_id="q001", status="success"):
    return {"dataset_name": dataset, "question_id": question_id, "source_document": "q1-26-2027(1).pdf", "actual": {"sources": [{"source_filename": "q1-26-2027.pdf"}], "retrieval_evaluation": {"hit_at_5": 1.0}, "answer_evaluation": {"correctness_score": 1.0, "groundedness_score": 1.0}}, "execution": {"status": status, "latency_ms": 10}, "trace": {"scoring_status": "scored" if status == "success" else "unscorable"}}


def test_aggregate_excludes_failed_answers_and_tracks_pending():
    result = aggregate_results([_row(), _row(question_id="q002", status="failed")], expected_questions=3)
    assert result["completed"] == 1
    assert result["failed"] == 1
    assert result["pending"] == 1
    assert result["metrics"]["answer_correctness"] == 1.0
    assert result["source_accuracy"] == 1.0


def test_load_current_results_filters_and_deduplicates(tmp_path):
    old, new = _row(), _row()
    new["execution"]["latency_ms"] = 20
    (tmp_path / "old.json").write_text(json.dumps(old), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps(new), encoding="utf-8")
    (tmp_path / "other.json").write_text(json.dumps(_row("Other", "q001")), encoding="utf-8")
    rows = load_current_results(tmp_path)
    assert len(rows) == 1
    assert rows[0]["execution"]["latency_ms"] == 20
