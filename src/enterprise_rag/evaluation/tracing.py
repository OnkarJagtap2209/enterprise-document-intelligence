"""Per-question evaluation traces."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    experiment: str
    question_id: str
    query: str
    chunk_ids: tuple[str, ...]
    metrics: dict[str, float]
    results: tuple[dict[str, Any], ...] = ()

def write_traces(path: str | Path, traces: list[EvaluationTrace]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(asdict(trace), sort_keys=True) + "\n" for trace in traces), encoding="utf-8")
