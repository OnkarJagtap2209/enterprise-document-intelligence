"""Phase 10 re-evaluation and baseline comparison utilities."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any, Iterable


def load_result_rows(directory: str | Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(row, dict) and row.get("question_id"):
            row["_path"] = str(path)
            rows.append(row)
    return rows


def result_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("dataset_name", "")), str(row.get("question_id", ""))


def identify_baseline(rows: Iterable[dict[str, Any]], expected_count: int = 60) -> list[dict[str, Any]]:
    """Select the newest complete unique-question run without changing its rows."""
    ordered = sorted(rows, key=lambda row: row.get("_path", ""), reverse=True)
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ordered:
        selected.setdefault(result_key(row), row)
    if len(selected) < expected_count:
        raise ValueError(f"baseline has {len(selected)} unique questions; expected {expected_count}")
    return [selected[key] for key in sorted(selected)]


def _status(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("execution", {}).get("status", "missing"))


def _scoring(row: dict[str, Any] | None) -> str:
    return str(((row or {}).get("trace") or {}).get("scoring_status", "unavailable"))


def _metrics(row: dict[str, Any] | None, section: str) -> dict[str, Any] | None:
    value = (row or {}).get("actual", {}).get(section)
    return value if isinstance(value, dict) else None


def _count(row: dict[str, Any] | None, field: str) -> int:
    value = (row or {}).get("actual", {}).get(field, [])
    return len(value) if isinstance(value, list) else 0


def _classification(base: dict[str, Any] | None, current: dict[str, Any] | None) -> str:
    bs, cs = _status(base), _status(current)
    if bs == "missing" or cs == "missing": return "unavailable"
    if bs != "success": return "baseline_failed"
    if cs != "success": return "phase10_failed"
    br, cr = _scoring(base), _scoring(current)
    if br == "unscorable" and cr != "unscorable": return "newly_scorable"
    if br != "unscorable" and cr == "unscorable": return "newly_unscorable"
    return "unchanged"


def _aggregate(rows: list[dict[str, Any]], section: str) -> dict[str, Any]:
    values: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        metrics = _metrics(row, section)
        if not metrics: continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool): values[key].append(float(value))
    return {key: {"mean": sum(vals) / len(vals), "valid": len(vals), "excluded": len(rows) - len(vals)} for key, vals in sorted(values.items())}


def _metric_deltas(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for key in sorted(set(base) & set(current)):
        if isinstance(base[key], dict) and isinstance(current[key], dict) and isinstance(base[key].get("mean"), (int, float)) and isinstance(current[key].get("mean"), (int, float)):
            deltas[key] = {"delta": current[key]["mean"] - base[key]["mean"], "baseline_valid": base[key]["valid"], "phase10_valid": current[key]["valid"]}
    return deltas


def build_comparison(baseline: Iterable[dict[str, Any]], current: Iterable[dict[str, Any]], expected_count: int = 60) -> dict[str, Any]:
    base_map = {result_key(row): row for row in baseline}
    current_map = {result_key(row): row for row in current}
    keys = sorted(set(base_map) | set(current_map))
    per_question = []
    for key in keys:
        base, now = base_map.get(key), current_map.get(key)
        per_question.append({
            "dataset_name": key[0], "question_id": key[1],
            "baseline_execution_status": _status(base), "phase10_execution_status": _status(now),
            "baseline_scoring_status": _scoring(base), "phase10_scoring_status": _scoring(now),
            "baseline_retrieved_source_count": _count(base, "sources"), "phase10_retrieved_source_count": _count(now, "sources"),
            "baseline_cited_source_count": _count(base, "cited_sources"), "phase10_cited_source_count": _count(now, "cited_sources"),
            "baseline_retrieval_metrics": _metrics(base, "retrieval_evaluation"), "phase10_retrieval_metrics": _metrics(now, "retrieval_evaluation"),
            "baseline_answer_metrics": _metrics(base, "answer_evaluation"), "phase10_answer_metrics": _metrics(now, "answer_evaluation"),
            "latency_difference_ms": ((now or {}).get("execution", {}).get("latency_ms") - (base or {}).get("execution", {}).get("latency_ms")) if isinstance((now or {}).get("execution", {}).get("latency_ms"), (int, float)) and isinstance((base or {}).get("execution", {}).get("latency_ms"), (int, float)) else None,
            "change_classification": _classification(base, now),
        })
    successful_base = [row for row in base_map.values() if _status(row) == "success"]
    successful_now = [row for row in current_map.values() if _status(row) == "success"]
    quota = [row for row in current_map.values() if "429" in str((row.get("execution") or {}).get("error", {})) or "RESOURCE_EXHAUSTED" in str((row.get("execution") or {}).get("error", {}))]
    retrieval_base = _aggregate(successful_base, "retrieval_evaluation")
    retrieval_current = _aggregate(successful_now, "retrieval_evaluation")
    answer_base = _aggregate(successful_base, "answer_evaluation")
    answer_current = _aggregate(successful_now, "answer_evaluation")
    failures = [{"dataset_name": key[0], "question_id": key[1], "phase10_status": _status(row), "stage": ((row.get("execution") or {}).get("error") or {}).get("stage"), "type": ((row.get("execution") or {}).get("error") or {}).get("type")} for key, row in sorted(current_map.items()) if _status(row) != "success"]
    return {
        "benchmark": {"expected_questions": expected_count, "baseline_questions": len(base_map), "phase10_questions": len(current_map), "comparison_status": "complete" if len(current_map) == expected_count else "partial"},
        "baseline": {"successful": len(successful_base), "failed": len(base_map) - len(successful_base)},
        "phase10": {"successful": len(successful_now), "failed": len(current_map) - len(successful_now), "gemini_quota_failures": len(quota)},
        "retrieval_comparison": {"baseline": retrieval_base, "phase10": retrieval_current, "deltas": _metric_deltas(retrieval_base, retrieval_current)},
        "answer_quality_comparison": {"baseline": answer_base, "phase10": answer_current, "deltas": _metric_deltas(answer_base, answer_current)},
        "evidence_comparison": {"baseline_retrieved_sources": sum(_count(row, "sources") for row in base_map.values()), "phase10_retrieved_sources": sum(_count(row, "sources") for row in current_map.values()), "baseline_cited_sources": sum(_count(row, "cited_sources") for row in base_map.values()), "phase10_cited_sources": sum(_count(row, "cited_sources") for row in current_map.values())},
        "per_question": per_question,
        "failure_comparison": failures,
        "limitations": ["Historical Phase 7 results are unchanged.", "Quota failures and missing metrics are excluded from quality aggregates.", "Newly scorable cases indicate improved observability, not automatically improved retrieval quality."],
        "interpretation": "Phase 9 changed evidence observability; this comparison does not attribute quality improvement without valid paired metrics.",
    }


def write_comparison(path: str | Path, report: dict[str, Any]) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
