"""Isolated Gemini generation service with injectable client."""
from dataclasses import dataclass
import json
import os
from typing import Any, Sequence
from .context_builder import ContextBuilder, ContextItem
from .prompts import build_grounded_prompt

class GeminiGenerationError(ValueError): pass
class CitationValidationError(ValueError): pass

@dataclass(frozen=True, slots=True)
class GenerationResult:
    answer: str; source_ids: tuple[str, ...]; model_name: str | None = None; usage: Any = None

class GeminiService:
    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash", client: Any | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not isinstance(model_name, str) or not model_name.strip(): raise GeminiGenerationError("model_name must not be empty")
        self.model_name, self._client = model_name, client

    def generate(self, prompt: str) -> GenerationResult:
        if not self.api_key: raise GeminiGenerationError("GEMINI_API_KEY is required")
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc: raise GeminiGenerationError(f"Could not initialize Gemini: {exc}") from exc
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "answer": {"type": "STRING"},
                        "source_ids": {"type": "ARRAY", "items": {"type": "STRING"}},
                    },
                    "required": ["answer", "source_ids"],
                },
            )
            response = self._client.models.generate_content(model=self.model_name, contents=prompt, config=config)
            text = getattr(response, "text", None)
        except Exception as exc: raise GeminiGenerationError(f"Gemini generation failed: {exc}") from exc
        if not isinstance(text, str) or not text.strip(): raise GeminiGenerationError("Gemini returned an empty answer")
        source_ids = _extract_source_ids(response, text)
        answer = _extract_answer(text) or text.strip()
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        return GenerationResult(answer, source_ids, self.model_name, usage)

class GroundedGenerator:
    def __init__(self, gemini: GeminiService, context_builder: ContextBuilder | None = None): self.gemini, self.context_builder = gemini, context_builder or ContextBuilder()
    def generate(self, question: str, results: Sequence[Any]) -> tuple[GenerationResult, tuple[ContextItem, ...]]:
        context = self.context_builder.build(results)
        generated = self.gemini.generate(build_grounded_prompt(question, context))
        valid = {item.chunk_id for item in context}
        canonical_ids = _canonical_source_ids(generated.source_ids, valid)
        unknown = set(canonical_ids) - valid
        if unknown: raise CitationValidationError("Unknown source IDs: " + ", ".join(sorted(unknown)))
        if canonical_ids != generated.source_ids:
            generated = GenerationResult(generated.answer, canonical_ids, generated.model_name, generated.usage)
        return generated, context


def _canonical_source_ids(source_ids: Sequence[str], valid: set[str]) -> tuple[str, ...]:
    """Resolve only unambiguous provider aliases to supplied context IDs."""
    resolved: list[str] = []
    for source_id in source_ids:
        if source_id in valid:
            resolved.append(source_id)
            continue
        aliases = [candidate for candidate in valid if candidate.endswith(source_id)]
        resolved.append(aliases[0] if len(aliases) == 1 else source_id)
    return tuple(resolved)

def _extract_source_ids(response: Any, text: str) -> tuple[str, ...]:
    for value in (getattr(response, "source_ids", None), getattr(response, "parsed", None)):
        if isinstance(value, dict): value = value.get("source_ids")
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return tuple(value)
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return ()
    if isinstance(payload, dict) and isinstance(payload.get("source_ids"), list):
        ids = payload["source_ids"]
        if all(isinstance(item, str) for item in ids): return tuple(ids)
    return ()

def _extract_answer(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    answer = payload.get("answer") if isinstance(payload, dict) else None
    return answer.strip() if isinstance(answer, str) and answer.strip() else None
