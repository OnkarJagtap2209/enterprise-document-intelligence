"""Relevance lookup helpers."""
from collections import defaultdict
from typing import Iterable
from .dataset import RelevanceJudgment
from enterprise_rag.routing import normalize_source_filename

def relevance_map(judgments: Iterable[RelevanceJudgment]) -> dict[str, dict[str, int]]:
    result = defaultdict(dict)
    for judgment in judgments:
        result[judgment.question_id][judgment.chunk_id] = judgment.relevance
    return {key: dict(value) for key, value in result.items()}

def source_is_relevant(source: dict, question: object) -> bool:
    """Match persisted source metadata against dataset evidence without inventing labels."""
    metadata = source.get("metadata", {}) if isinstance(source, dict) else {}
    filename = source.get("source_filename") if isinstance(source, dict) else None
    expected_document = getattr(question, "document", None)
    if expected_document and filename:
        same_document = normalize_source_filename(PathLikeName(filename)) == normalize_source_filename(PathLikeName(expected_document))
        if not same_document:
            return False
        pages = set(getattr(question, "relevant_pages", ()))
        page = source.get("page_start") if isinstance(source, dict) else None
        if not pages or page in pages:
            return True
    text = str(source.get("content", "")) if isinstance(source, dict) else ""
    return bool(text and any(target.casefold() in text.casefold() for target in getattr(question, "retrieval_targets", ())))

def PathLikeName(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].casefold()
