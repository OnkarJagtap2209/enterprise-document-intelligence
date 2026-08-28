"""Isolated Gemini generation service with injectable client."""
from dataclasses import dataclass
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
            response = self._client.models.generate_content(model=self.model_name, contents=prompt)
            text = getattr(response, "text", None)
        except Exception as exc: raise GeminiGenerationError(f"Gemini generation failed: {exc}") from exc
        if not isinstance(text, str) or not text.strip(): raise GeminiGenerationError("Gemini returned an empty answer")
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        return GenerationResult(text.strip(), (), self.model_name, usage)

class GroundedGenerator:
    def __init__(self, gemini: GeminiService, context_builder: ContextBuilder | None = None): self.gemini, self.context_builder = gemini, context_builder or ContextBuilder()
    def generate(self, question: str, results: Sequence[Any]) -> tuple[GenerationResult, tuple[ContextItem, ...]]:
        context = self.context_builder.build(results)
        generated = self.gemini.generate(build_grounded_prompt(question, context))
        valid = {item.chunk_id for item in context}
        unknown = set(generated.source_ids) - valid
        if unknown: raise CitationValidationError("Unknown source IDs: " + ", ".join(sorted(unknown)))
        return GenerationResult(generated.answer, tuple(generated.source_ids)), context
