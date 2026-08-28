"""Strict validation of generated source identifiers."""
from typing import Iterable
from .context_builder import ContextItem
from .gemini import CitationValidationError

def validate_sources(source_ids: Iterable[str], context: Iterable[ContextItem]) -> tuple[ContextItem, ...]:
    by_id = {item.chunk_id: item for item in context}; ids = tuple(source_ids)
    unknown = [item for item in ids if item not in by_id]
    if unknown: raise CitationValidationError("Unknown source IDs: " + ", ".join(unknown))
    return tuple(by_id[item] for item in ids)
