"""Reproducible experiment execution over retriever interfaces."""
from dataclasses import dataclass
from typing import Any, Sequence
from .dataset import EvaluationQuestion, RelevanceJudgment
from .relevance import relevance_map
from .retrieval_metrics import compute_metrics
from .tracing import EvaluationTrace

@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    name: str
    top_k: int = 5
    version: str = "1"
    chunking: str = "structure_aware"
    retrieval: str = "semantic"

@dataclass(frozen=True, slots=True)
class ExperimentRun:
    spec: ExperimentSpec
    traces: tuple[EvaluationTrace, ...]
    summary: dict[str, float]

class ExperimentRunner:
    def __init__(self, retriever: Any, spec: ExperimentSpec, reranker: Any | None = None):
        if spec.top_k <= 0: raise ValueError("top_k must be greater than zero")
        self.retriever, self.spec, self.reranker = retriever, spec, reranker

    def run(self, questions: Sequence[EvaluationQuestion], judgments: Sequence[RelevanceJudgment]) -> ExperimentRun:
        lookup = relevance_map(judgments); traces = []
        for question in questions:
            depth = getattr(self.reranker, "candidate_depth", self.spec.top_k)
            results = tuple(self.retriever.retrieve(question.query, top_k=depth))
            if self.reranker is not None:
                results = tuple(self.reranker.rerank(question.query, results, top_k=self.spec.top_k))
            ids = tuple(result.chunk_id for result in results)
            trace = EvaluationTrace(self.spec.name, question.question_id, question.query, ids, compute_metrics(ids, lookup.get(question.question_id, {}), self.spec.top_k), tuple({"chunk_id": result.chunk_id, "rank": result.rank} for result in results))
            traces.append(trace)
        summary = {key: sum(trace.metrics[key] for trace in traces) / len(traces) for key in traces[0].metrics} if traces else {}
        return ExperimentRun(self.spec, tuple(traces), summary)
