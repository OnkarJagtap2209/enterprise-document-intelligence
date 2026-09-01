"""Execute and persist one raw evaluation observation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import inspect
from time import perf_counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .dataset import EvaluationQuestion, EvaluationDataset
from .relevance import source_is_relevant
from .retrieval_metrics import compute_retrieval_metrics
from .answer_metrics import evaluate_answer

@dataclass(frozen=True, slots=True)
class EvaluationExecution:
    dataset_name: str
    source_document: str
    question_id: str
    question: str
    expected: dict[str, Any]
    actual: dict[str, Any]
    execution: dict[str, Any]
    trace: dict[str, Any] | None = None

def execute_one(dataset: EvaluationDataset, question: EvaluationQuestion, rag_service: Any) -> EvaluationExecution:
    started = perf_counter(); started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid4())
    try:
        query_fn = rag_service.query
        if "source_filename" in inspect.signature(query_fn).parameters:
            outcome = query_fn(question.query, source_filename=dataset.source_document.get("filename"))
        else:
            outcome = query_fn(question.query)
        cited = [_source(item) for item in getattr(outcome, "sources", ())]
        retrieved = [_source(item) for item in getattr(outcome, "retrieved_sources", ())]
        sources = retrieved or cited
        actual = {"answer": getattr(outcome, "answer", None), "sources": sources, "cited_sources": cited}
        execution = {"status": "success", "latency_ms": (perf_counter() - started) * 1000, "request_id": getattr(outcome, "request_id", None), "error": None}
    except Exception as exc:
        actual = {"answer": None, "sources": []}
        execution = {"status": "failed", "latency_ms": (perf_counter() - started) * 1000, "request_id": None, "error": {"stage": _failure_stage(exc), "type": type(exc).__name__, "message": str(exc)}}
    expected = {"answer": question.expected_answer, "facts": list(question.expected_facts), "retrieval_targets": list(question.retrieval_targets), "relevant_pages": list(question.relevant_pages), "evidence": question.evidence}
    trace = {"run_id": run_id, "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(), "retrieved_result_count": len(actual["sources"]), "source_count": len(actual["sources"]), "answer_present": bool(actual["answer"]), "scoring_status": "unscored"}
    return EvaluationExecution(dataset.dataset_name, dataset.source_document["filename"], question.question_id, question.query, expected, actual, execution, trace)

def evaluate_one(dataset: EvaluationDataset, question: EvaluationQuestion, rag_service: Any) -> EvaluationExecution:
    result = execute_one(dataset, question, rag_service)
    if result.execution["status"] != "success":
        return result
    sources = result.actual["sources"]
    relevant = {source["chunk_id"] for source in sources if source_is_relevant(source, question)}
    retrieved = [source["chunk_id"] for source in sources if source.get("chunk_id")]
    retrieval = compute_retrieval_metrics(retrieved, relevant) if retrieved and relevant else {"status": "unscorable", "reason": "retrieved evidence or relevance ground truth unavailable"}
    answer = evaluate_answer(result.actual["answer"], question.expected_facts, sources, question.evidence)
    retrieval_scored = retrieval.get("status") != "unscorable"
    answer_scored = answer.evaluator_status == "complete"
    if retrieval_scored and answer_scored:
        scoring = "scored"
    elif not retrieval_scored and not answer_scored:
        scoring = "unscorable"
    else:
        scoring = "partially_scorable"
    actual = dict(result.actual); actual["retrieval_evaluation"] = retrieval; actual["answer_evaluation"] = asdict(answer)
    trace = dict(result.trace or {}); trace["scoring_status"] = scoring
    return EvaluationExecution(result.dataset_name, result.source_document, result.question_id, result.question, result.expected, actual, result.execution, trace)

def summarize_results(results: list[EvaluationExecution]) -> dict[str, Any]:
    successful = [r for r in results if r.execution.get("status") == "success"]
    latencies = [float(r.execution["latency_ms"]) for r in successful if isinstance(r.execution.get("latency_ms"), (int, float))]
    scoring = [r.trace.get("scoring_status") if r.trace else "unscorable" for r in successful]
    summary: dict[str, Any] = {"total_questions": len(results), "successful_executions": len(successful), "failed_executions": len(results)-len(successful), "execution_success_rate": len(successful)/len(results) if results else 0.0, "average_latency_ms": sum(latencies)/len(latencies) if latencies else None, "scored_cases": scoring.count("scored"), "partially_scorable_cases": scoring.count("partially_scorable"), "unscorable_cases": scoring.count("unscorable")}
    for section, prefix in (("retrieval_evaluation", "retrieval_"), ("answer_evaluation", "answer_")):
        values: dict[str, list[float]] = {}
        for result in successful:
            for key, value in result.actual.get(section, {}).items():
                if isinstance(value, (int, float)): values.setdefault(prefix + key, []).append(float(value))
        summary.update({key: sum(items)/len(items) for key, items in values.items()})
    return summary

def persist_execution(result: EvaluationExecution, directory: str | Path) -> Path:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{stamp}_{result.question_id}_{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path

def _source(source: Any) -> dict[str, Any]:
    metadata = dict(getattr(source, "metadata", {}) or {})
    return {"chunk_id": getattr(source, "chunk_id", None), "document_id": getattr(source, "document_id", None), "source_filename": metadata.get("source_filename"), "page_start": metadata.get("page_start"), "page_end": metadata.get("page_end"), "content": getattr(source, "content", None), "metadata": metadata, "provenance": list(getattr(source, "provenance", ()))}

def _failure_stage(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    for token, stage in (("citation", "citation"), ("generation", "generation"), ("retrieval", "retrieval"), ("routing", "routing"), ("dataset", "dataset"), ("persist", "persistence")):
        if token in name: return stage
    return "unknown"
