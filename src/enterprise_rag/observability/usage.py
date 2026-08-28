"""Honest model usage extraction."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class UsageInfo:
    input_tokens: int | None = None; output_tokens: int | None = None; total_tokens: int | None = None

    @classmethod
    def from_response(cls, response: Any) -> "UsageInfo":
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if usage is None: return cls()
        def get(*names):
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if isinstance(value, int) and not isinstance(value, bool): return value
            return None
        return cls(get("prompt_token_count", "input_tokens"), get("candidates_token_count", "output_tokens"), get("total_token_count", "total_tokens"))
