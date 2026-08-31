"""Validated loaders for versioned evaluation question datasets."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

class EvaluationDatasetError(ValueError): pass

@dataclass(frozen=True, slots=True)
class EvaluationQuestion:
    question_id: str
    query: str
    version: str = "1"
    expected_answer: str | None = None
    expected_facts: tuple[str, ...] = ()
    retrieval_targets: tuple[str, ...] = ()
    document: str | None = None
    relevant_pages: tuple[int, ...] = ()
    evidence: str | None = None
    question_type: str | None = None
    raw: dict[str, Any] | None = None

@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    question_id: str
    chunk_id: str
    relevance: int

@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    path: Path
    dataset_version: str
    dataset_name: str
    source_document: dict[str, Any]
    notes: tuple[str, ...]
    questions: tuple[EvaluationQuestion, ...]

def load_dataset(path: str | Path) -> EvaluationDataset:
    path = Path(path); payload = _load(path)
    if not isinstance(payload, dict): raise EvaluationDatasetError(f"{path.name}: root must be an object")
    version, name, source, rows = payload.get("dataset_version"), payload.get("dataset_name"), payload.get("source_document"), payload.get("questions")
    if not isinstance(version, str) or not version: raise EvaluationDatasetError(f"{path.name}: missing dataset_version")
    if not isinstance(name, str) or not name: raise EvaluationDatasetError(f"{path.name}: missing dataset_name")
    if not isinstance(source, dict) or not isinstance(source.get("filename"), str): raise EvaluationDatasetError(f"{path.name}: missing source_document.filename")
    if not isinstance(rows, list) or not rows: raise EvaluationDatasetError(f"{path.name}: questions must be a non-empty list")
    questions = tuple(_question(row, path, version) for row in rows)
    ids = [q.question_id for q in questions]
    if len(set(ids)) != len(ids): raise EvaluationDatasetError(f"{path.name}: duplicate question_id")
    return EvaluationDataset(path.resolve(), version, name, dict(source), tuple(str(n) for n in payload.get("notes", [])), questions)

def discover_datasets(directory: str | Path) -> tuple[EvaluationDataset, ...]:
    paths = sorted(Path(directory).glob("*.json"))
    if not paths: raise EvaluationDatasetError(f"No evaluation datasets found in {directory}")
    datasets = tuple(load_dataset(path) for path in paths)
    return datasets

def load_questions(path: str | Path) -> tuple[EvaluationQuestion, ...]: return load_dataset(path).questions

def load_relevance(path: str | Path) -> tuple[RelevanceJudgment, ...]:
    payload = _load(path); rows = payload.get("judgments", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list): raise EvaluationDatasetError(f"{Path(path).name}: judgments must be a list")
    try: return tuple(RelevanceJudgment(str(r["question_id"]), str(r["chunk_id"]), int(r["relevance"])) for r in rows)
    except (KeyError, TypeError, ValueError) as exc: raise EvaluationDatasetError(f"{Path(path).name}: malformed relevance judgment") from exc

def _question(row: Any, path: Path, version: str) -> EvaluationQuestion:
    if not isinstance(row, dict) or not isinstance(row.get("question_id"), str) or not isinstance(row.get("question"), str): raise EvaluationDatasetError(f"{path.name}: each question needs question_id and question")
    pages, facts, targets = row.get("relevant_pages", []), row.get("expected_facts", []), row.get("retrieval_targets", [])
    if not all(isinstance(v, int) and v > 0 for v in pages): raise EvaluationDatasetError(f"{path.name}: relevant_pages must contain positive integers")
    if not all(isinstance(v, str) for v in (*facts, *targets)): raise EvaluationDatasetError(f"{path.name}: expected_facts/retrieval_targets must contain strings")
    return EvaluationQuestion(row["question_id"], row["question"], str(row.get("version", version)), row.get("expected_answer"), tuple(facts), tuple(targets), row.get("document"), tuple(pages), row.get("evidence"), row.get("question_type"), dict(row))

def _load(path: str | Path) -> Any:
    path = Path(path)
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc: raise EvaluationDatasetError(f"{path.name}: could not load JSON: {exc}") from exc
