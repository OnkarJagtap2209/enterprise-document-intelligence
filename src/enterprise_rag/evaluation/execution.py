"""Execute and persist one raw evaluation observation."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from time import perf_counter
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .dataset import EvaluationQuestion, EvaluationDataset

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
        outcome = rag_service.query(question.query)
        sources = [_source(item) for item in getattr(outcome, "sources", ())]
        actual = {"answer": getattr(outcome, "answer", None), "sources": sources}
        execution = {"status": "success", "latency_ms": (perf_counter() - started) * 1000, "request_id": getattr(outcome, "request_id", None), "error": None}
    except Exception as exc:
        actual = {"answer": None, "sources": []}
        execution = {"status": "failed", "latency_ms": (perf_counter() - started) * 1000, "request_id": None, "error": {"stage": _failure_stage(exc), "type": type(exc).__name__, "message": str(exc)}}
    expected = {"answer": question.expected_answer, "facts": list(question.expected_facts), "retrieval_targets": list(question.retrieval_targets), "relevant_pages": list(question.relevant_pages), "evidence": question.evidence}
    trace = {"run_id": run_id, "started_at": started_at, "ended_at": datetime.now(timezone.utc).isoformat(), "retrieved_result_count": len(actual["sources"]), "source_count": len(actual["sources"]), "answer_present": bool(actual["answer"]), "scoring_status": "unscored"}
    return EvaluationExecution(dataset.dataset_name, dataset.source_document["filename"], question.question_id, question.query, expected, actual, execution, trace)

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
