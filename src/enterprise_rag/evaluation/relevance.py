"""Relevance lookup helpers."""
from collections import defaultdict
from typing import Iterable
from .dataset import RelevanceJudgment

def relevance_map(judgments: Iterable[RelevanceJudgment]) -> dict[str, dict[str, int]]:
    result = defaultdict(dict)
    for judgment in judgments:
        result[judgment.question_id][judgment.chunk_id] = judgment.relevance
    return {key: dict(value) for key, value in result.items()}
