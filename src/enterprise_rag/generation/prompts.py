"""Grounded prompt construction."""
from typing import Sequence
from .context_builder import ContextItem

def build_grounded_prompt(question: str, context: Sequence[ContextItem]) -> str:
    if not isinstance(question, str) or not question.strip(): raise ValueError("question must be non-empty")
    evidence = "\n\n".join(f"[{item.chunk_id}] ({item.document_id})\n{item.content}" for item in context)
    if not evidence: evidence = "[NO_EVIDENCE]\nNo evidence was retrieved."
    return "SYSTEM:\nAnswer only from supplied evidence. Do not invent facts or citations. State when evidence is insufficient.\n\nUSER QUESTION:\n" + question.strip() + "\n\nSUPPLIED EVIDENCE:\n" + evidence
