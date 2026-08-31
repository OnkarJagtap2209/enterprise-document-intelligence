"""Deterministic answer correctness and groundedness checks."""
from dataclasses import asdict, dataclass
import re
from typing import Any, Sequence

@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    correctness_status: str
    correctness_score: float | None
    supported_expected_facts: tuple[str, ...]
    missing_expected_facts: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    groundedness_status: str
    groundedness_score: float | None
    reason: str
    evaluator_status: str = "complete"
    unscorable_reason: str | None = None
    errors: tuple[str, ...] = ()

def evaluate_answer(answer: str | None, expected_facts: Sequence[str], sources: Sequence[dict[str, Any]], evidence: str | None = None) -> AnswerEvaluation:
    facts = tuple(str(f) for f in expected_facts)
    if not isinstance(answer, str) or not answer.strip():
        return AnswerEvaluation("unscorable", None, (), facts, (), "unscorable", None, "Generated answer is empty.", "unscorable", "empty answer")
    if not facts:
        return AnswerEvaluation("unscorable", None, (), (), (), "unscorable", None, "No expected facts are available.", "unscorable", "missing expected facts")
    supported = tuple(f for f in facts if _fact_present(f, answer))
    missing = tuple(f for f in facts if f not in supported)
    score = len(supported) / len(facts)
    status = "correct" if not missing else ("partially_correct" if supported else "incorrect")
    evidence_text = "\n".join(str(s.get("content", "")) for s in sources if isinstance(s, dict))
    if evidence:
        evidence_text += "\n" + evidence
    if not evidence_text.strip():
        return AnswerEvaluation(status, score, supported, missing, (), "unscorable", None, "No retrieved evidence is available.", "unscorable", "no retrieved evidence")
    grounded = tuple(f for f in supported if _fact_present(f, evidence_text))
    gscore = len(grounded) / len(facts)
    gstatus = "grounded" if len(grounded) == len(facts) else ("partially_grounded" if grounded else "ungrounded")
    return AnswerEvaluation(status, score, supported, missing, (), gstatus, gscore, "Deterministic fact and evidence comparison.")

def _fact_present(fact: str, text: str) -> bool:
    key, _, value = fact.partition("=")
    target = value or key
    normalized = re.sub(r"[^a-z0-9.%]", "", target.casefold())
    candidate = re.sub(r"[^a-z0-9.%]", "", text.casefold())
    return bool(normalized) and normalized in candidate

__all__ = ["AnswerEvaluation", "evaluate_answer"]
