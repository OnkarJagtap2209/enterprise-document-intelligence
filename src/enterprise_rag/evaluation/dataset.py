"""Versioned evaluation questions and relevance judgments."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    query: str
    version: str = "1"

@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    question_id: str
    chunk_id: str
    relevance: int

def load_questions(path: str | Path) -> tuple[EvaluationQuestion, ...]:
    payload = _load(path)
    rows = payload.get("questions", payload) if isinstance(payload, dict) else payload
    return tuple(EvaluationQuestion(str(r["question_id"]), str(r["query"]), str(r.get("version", payload.get("version", "1") if isinstance(payload, dict) else "1"))) for r in rows)

def load_relevance(path: str | Path) -> tuple[RelevanceJudgment, ...]:
    payload = _load(path)
    rows = payload.get("judgments", payload) if isinstance(payload, dict) else payload
    return tuple(RelevanceJudgment(str(r["question_id"]), str(r["chunk_id"]), int(r["relevance"])) for r in rows)

def _load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
