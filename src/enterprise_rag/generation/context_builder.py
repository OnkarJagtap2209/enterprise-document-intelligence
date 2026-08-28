"""Deterministic bounded context construction."""
from dataclasses import dataclass
from typing import Any, Sequence

@dataclass(frozen=True, slots=True)
class ContextItem:
    chunk_id: str; document_id: str; content: str; metadata: dict[str, Any]; provenance: tuple[dict[str, Any], ...]

class ContextBuilder:
    def __init__(self, max_chars: int = 12000):
        if not isinstance(max_chars, int) or max_chars <= 0: raise ValueError("max_chars must be greater than zero")
        self.max_chars = max_chars

    def build(self, results: Sequence[Any]) -> tuple[ContextItem, ...]:
        items = []; seen = set(); used = 0
        for result in results:
            if result.chunk_id in seen: continue
            content = getattr(result, "content", "")
            if not isinstance(content, str) or not content.strip(): continue
            if used + len(content) > self.max_chars: break
            item = ContextItem(result.chunk_id, result.document_id, content, dict(result.metadata), tuple(getattr(result, "provenance", ())))
            items.append(item); seen.add(item.chunk_id); used += len(content)
        return tuple(items)
