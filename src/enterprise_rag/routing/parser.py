"""Deterministic extraction of explicit metadata constraints."""
from dataclasses import dataclass
import re

@dataclass(frozen=True, slots=True)
class QueryConstraints:
    document_id: str | None = None
    source_filename: str | None = None
    content_type: str | None = None
    year: int | None = None

def parse_query(query: str) -> tuple[QueryConstraints, str | None]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = query.strip(); years = [int(y) for y in re.findall(r"\b(?:20|19)\d{2}\b", text)]
    if len(set(years)) > 1:
        return QueryConstraints(), "Which document year should I use?"
    filename = None
    match = re.search(r"\b[\w.-]+\.pdf\b", text, re.I)
    if match: filename = match.group(0)
    content_type = None
    if re.search(r"\btable\b", text, re.I): content_type = "table"
    elif re.search(r"\b(?:paragraph|text)\b", text, re.I): content_type = "text"
    return QueryConstraints(source_filename=filename, content_type=content_type, year=years[0] if years else None), None
