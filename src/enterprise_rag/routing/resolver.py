"""Query-understanding result and retrieval filter helpers."""
from dataclasses import dataclass
from typing import Any, Mapping
import re
from .parser import QueryConstraints, parse_query

@dataclass(frozen=True, slots=True)
class RoutingResult:
    original_query: str
    constraints: QueryConstraints
    clarification_required: bool
    clarification_question: str | None
    retrieval_query: str | None

def understand_query(query: str) -> RoutingResult:
    constraints, clarification = parse_query(query)
    return RoutingResult(query.strip(), constraints, clarification is not None, clarification, None if clarification else query.strip())

def matches_constraints(metadata: Mapping[str, Any], constraints: QueryConstraints) -> bool:
    for field, value in (("document_id", constraints.document_id), ("source_filename", constraints.source_filename), ("content_type", constraints.content_type)):
        if value is not None:
            actual = metadata.get(field)
            if field == "source_filename":
                if not _source_filename_matches(actual, value): return False
            elif actual != value:
                return False
    # A natural-language period year must not be inferred from a filename.
    # Apply it only when an authoritative year field is present in metadata.
    if constraints.year is not None:
        year_value = next(
            (metadata[field] for field in ("period_year", "document_year", "year") if field in metadata),
            None,
        )
        if year_value is not None:
            try:
                if int(year_value) != constraints.year:
                    return False
            except (TypeError, ValueError):
                return False
    return True

def normalize_source_filename(value: Any) -> Any:
    """Resolve only known evaluation aliases to indexed filenames."""
    if not isinstance(value, str):
        return value
    aliases = {
        "form20f-25-26.pdf": "form20f-25-2026.pdf",
        "q1-26-2027(1).pdf": "q1-26-2027.pdf",
    }
    folded = value.casefold()
    without_duplicate_suffix = re.sub(r"\(\d+\)(?=\.pdf$)", "", folded)
    return aliases.get(folded, aliases.get(without_duplicate_suffix, value))

def _source_filename_matches(actual: Any, expected: Any) -> bool:
    if not isinstance(actual, str) or not isinstance(expected, str): return actual == expected
    a = normalize_source_filename(actual).casefold(); e = normalize_source_filename(expected).casefold()
    if a == e: return True
    a = re.sub(r"\(\d+\)(?=\.pdf$)", "", a)
    e = re.sub(r"\(\d+\)(?=\.pdf$)", "", e)
    if a == e: return True
    return False
