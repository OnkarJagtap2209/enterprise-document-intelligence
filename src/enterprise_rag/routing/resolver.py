"""Query-understanding result and retrieval filter helpers."""
from dataclasses import dataclass
from typing import Any, Mapping
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
        if value is not None and metadata.get(field) != value: return False
    if constraints.year is not None and str(constraints.year) not in str(metadata.get("source_filename", "")): return False
    return True
