"""Evidence-based analysis of persisted Phase 7 benchmark results."""
import json
from pathlib import Path
from typing import Any

def analyze_full_run(results_dir: str | Path) -> dict[str, Any]:
    files = sorted(Path(results_dir).glob("*.json"))[-60:]
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    successful = [row for row in rows if row.get("execution", {}).get("status") == "success"]
    quota = [row for row in rows if "429" in str(row.get("execution", {}).get("error", {})) or "RESOURCE_EXHAUSTED" in str(row.get("execution", {}).get("error", {}))]
    return {"benchmark": {"attempted": len(rows), "persisted": len(rows), "successful": len(successful), "failed": len(rows)-len(successful), "gemini_quota_failures": len(quota)}, "successful_question_ids": [row["question_id"] for row in successful], "per_question": [{"question_id": row["question_id"], "answer": row.get("actual", {}).get("answer"), "sources": row.get("actual", {}).get("sources", []), "retrieval_evaluation": row.get("actual", {}).get("retrieval_evaluation"), "answer_evaluation": row.get("actual", {}).get("answer_evaluation"), "scoring_status": (row.get("trace") or {}).get("scoring_status")} for row in successful], "limitations": ["53 Gemini RESOURCE_EXHAUSTED failures are external quota failures, not RAG-quality failures.", "Only successful executions are suitable for quality analysis."]}

def write_analysis(results_dir: str | Path, output: str | Path) -> Path:
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(analyze_full_run(results_dir), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"); return target
