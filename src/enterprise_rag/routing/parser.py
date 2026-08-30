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
    filename = None
    match = re.search(r"\b[\w.-]+\.pdf\b", text, re.I)
    if match: filename = match.group(0)
    document_year_intent = bool(
        re.search(r"\b(?:annual\s+)?reports?|documents?|filings?\b", text, re.I)
    )
    if len(set(years)) > 1 and document_year_intent and filename is None:
        return QueryConstraints(), "Which document year should I use?"
    content_type = None
    if re.search(r"\btable\b", text, re.I): content_type = "table"
    elif re.search(r"\b(?:paragraph|text)\b", text, re.I): content_type = "text"
    year = years[0] if len(set(years)) == 1 else None
    return QueryConstraints(source_filename=filename, content_type=content_type, year=year), None
