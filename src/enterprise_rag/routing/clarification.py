from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    question: str

def needs_clarification(question: str | None) -> bool:
    return bool(question)
